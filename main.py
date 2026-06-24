"""
main.py
=======
电商智能客服 NLU 意图识别引擎 — 统一命令行入口

命令:
    python main.py train [OPTIONS]    训练/评估意图分类模型
    python main.py serve [OPTIONS]    启动 Flask 推理服务
    python main.py predict [OPTIONS]  命令行单条意图识别
    python main.py info               显示系统信息与配置

示例:
    # 训练逻辑回归模型
    python main.py train --model lr

    # 多模型对比实验
    python main.py train --all --compare

    # 启动推理服务
    python main.py serve --host 0.0.0.0 --port 5000

    # 命令行意图识别
    python main.py predict --text "我的订单什么时候发货"

    # 查看系统信息
    python main.py info
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("iot-qa-main")


# ============================================================
# 子命令: train
# ============================================================

def _cmd_train(args: argparse.Namespace) -> None:
    """训练模型子命令。

    流水线: 加载数据 → 清洗 → 特征提取 → 标签编码 → 训练 → 评估 → 保存
    """
    logger.info("=" * 60)
    logger.info("开始模型训练流水线")
    logger.info("=" * 60)

    # 1. 加载数据
    from src.data_pipeline.loader import DataLoader
    loader = DataLoader(settings.RAW_DATA_PATH)
    df = loader.load()
    logger.info(f"[Step 1/7] 数据加载完成: {len(df)} 条")

    # 2. 清洗数据
    from src.data_pipeline.cleaner import DataCleaner
    cleaner = DataCleaner()
    texts, labels = cleaner.run(df)
    logger.info(cleaner.summary())
    logger.info(f"[Step 2/7] 数据清洗完成: {len(texts)} 条")

    # 3. 划分数据集
    from src.training.trainer import Trainer
    trainer = Trainer(classifier_type=_resolve_classifier(args))

    X_train, X_val, X_test, y_train, y_val, y_test = Trainer.split_data(
        texts, labels,
        test_size=settings.TRAIN_TEST_SIZE,
        val_size=settings.TRAIN_VAL_SIZE,
        random_state=settings.TRAIN_RANDOM_STATE,
    )
    logger.info(f"[Step 3/7] 数据集划分完成")

    # 4. 特征提取
    from src.features.vectorizer import VectorizerManager
    vectorizer = VectorizerManager(settings.tfidf_config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    logger.info(f"[Step 4/7] TF-IDF 特征提取完成 — 维度: {vectorizer.n_features}")

    # 5. 标签编码
    trainer.fit_binarizer(y_train)
    assert trainer.mlb is not None
    y_train_mat = trainer.mlb.transform(y_train)
    y_val_mat = trainer.mlb.transform(y_val)
    y_test_mat = trainer.mlb.transform(y_test)
    logger.info(f"[Step 5/7] 标签编码完成 — 类别数: {len(trainer.mlb.classes_)}")

    # 6. 多模型对比（可选）
    if args.all or args.compare:
        logger.info(f"\n{'='*60}")
        logger.info("[Step 6/7] 多模型对比实验")
        logger.info(f"{'='*60}")
        from src.training.trainer import Trainer as T
        results_df, _ = T.compare_models(
            X_train_vec, y_train_mat, X_test_vec, y_test_mat,
            use_grid_search=args.grid_search,
            label_names=list(trainer.mlb.classes_),
        )
        best_type = results_df.loc[0, "_type"] if "_type" in results_df.columns else _resolve_classifier(args)
        logger.info(f"\n最佳模型: {results_df.loc[0, 'model']} (f1_micro={results_df.loc[0, 'f1_micro']:.4f})")
    else:
        best_type = _resolve_classifier(args)

    # 7. 训练最终模型 + 评估
    logger.info(f"\n{'='*60}")
    logger.info(f"[Step 7/7] 训练最终模型: {best_type}")
    logger.info(f"{'='*60}")

    final_trainer = Trainer(classifier_type=best_type)
    final_trainer.fit_binarizer(y_train)

    if args.grid_search:
        best_model, tune_result = final_trainer.grid_search_train(
            X_train_vec, y_train_mat, cv=args.cv,
        )
        tune_result = tune_result  # noqa: F841 — 可用于后续展示
    else:
        final_trainer.mlb = trainer.mlb
        final_trainer.train(X_train_vec, y_train_mat, X_val_vec, y_val_mat)

    # 评估
    metrics = final_trainer.evaluate(X_test_vec, y_test_mat)

    # 打印摘要
    print("\n" + "=" * 60)
    print("训练总结")
    print("=" * 60)
    print(f"最佳模型: {best_type}")
    if args.grid_search and 'tune_result' in dir():
        print(f"最佳参数: {tune_result['best_params']}")
        print(f"交叉验证 F1-micro: {tune_result['best_score']:.4f}")
    print(f"测试集 Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"测试集 F1-micro:       {metrics['f1_micro']:.4f}")
    print(f"测试集 F1-macro:       {metrics['f1_macro']:.4f}")
    print(f"测试集 Hamming Loss:   {metrics['hamming_loss']:.4f}")
    print(f"测试集 Jaccard-micro:  {metrics['jaccard_micro']:.4f}")
    print("=" * 60)

    # 8. 保存模型
    final_trainer.save_artifacts(
        vectorizer=vectorizer.vectorizer,
        model_path=settings.model_path,
        vectorizer_path=settings.vectorizer_path,
        mlb_path=settings.mlb_path,
    )
    logger.info("训练完成！模型文件已保存到 output/models/")

    # Per-class 评估
    print("\n" + final_trainer.evaluator.report(metrics))


def _resolve_classifier(args: argparse.Namespace) -> str:
    """解析 --model 参数为分类器注册表 key。"""
    model_map = {
        "lr": "logistic_regression",
        "rf": "random_forest",
        "svm": "svm",
        "all": "logistic_regression",
    }
    return model_map.get(args.model, "logistic_regression")


# ============================================================
# 子命令: serve
# ============================================================

def _cmd_serve(args: argparse.Namespace) -> None:
    """启动 Flask 推理服务。"""
    from app import app

    host = args.host or settings.FLASK_HOST
    port = args.port or settings.FLASK_PORT
    debug = args.debug or settings.FLASK_DEBUG

    logger.info(f"启动 Flask 服务: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


# ============================================================
# 子命令: predict (CLI 推理)
# ============================================================

def _cmd_predict(args: argparse.Namespace) -> None:
    """命令行单次推理。"""
    from src.inference.model_manager import ModelManager
    from src.inference.predictor import Predictor

    manager = ModelManager()
    manager.load()

    if not manager.is_ready():
        logger.error("模型未加载，请先运行 python main.py train")
        sys.exit(1)

    predictor = Predictor(manager)
    results = predictor.predict(
        text=args.text,
        threshold=args.threshold,
        max_labels=args.max_labels,
    )

    if not results:
        results = Predictor.build_fallback_result()

    print(f"\n输入: {args.text}")
    print(f"阈值: {args.threshold}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        action = r.get("action", {})
        print(f"  #{i} [{r['label']}] (置信度: {r['score']:.4f})")
        print(f"     路由: {action.get('route', 'N/A')} -> {action.get('target', 'N/A')}")
    print()


# ============================================================
# 子命令: info
# ============================================================

def _cmd_info(args: argparse.Namespace) -> None:
    """显示系统信息与配置。"""
    print("\n" + "=" * 50)
    print("电商智能客服 NLU 意图识别引擎 — 系统信息")
    print("=" * 50)
    print(f"Python 版本:     {sys.version.split()[0]}")
    print(f"项目根目录:     {settings.PROJECT_ROOT}")
    print(f"数据目录:       {settings.DATA_DIR}")
    print(f"模型目录:       {settings.MODEL_DIR}")
    print(f"日志目录:       {settings.LOG_DIR}")
    print(f"原始数据路径:   {settings.RAW_DATA_PATH}")
    print(f"清洗数据路径:   {settings.CLEANED_DATA_PATH}")
    print("-" * 50)
    print(f"TF-IDF 特征数:  {settings.TFIDF_MAX_FEATURES}")
    print(f"N-gram 范围:    ({settings.TFIDF_NGRAM_MIN}, {settings.TFIDF_NGRAM_MAX})")
    print(f"分析器:         {settings.TFIDF_ANALYZER}")
    print("-" * 50)
    print(f"训练测试比例:   {settings.TRAIN_TEST_SIZE}")
    print(f"验证集比例:     {settings.TRAIN_VAL_SIZE}")
    print(f"随机种子:       {settings.TRAIN_RANDOM_STATE}")
    print("-" * 50)
    print(f"推理默认阈值:   {settings.INFERENCE_THRESHOLD}")
    print(f"最大返回标签:   {settings.INFERENCE_MAX_LABELS}")
    print(f"批量推理上限:   {settings.INFERENCE_BATCH_SIZE}")
    print("-" * 50)
    print(f"Flask 默认地址: {settings.FLASK_HOST}:{settings.FLASK_PORT}")
    print(f"日志等级:       {settings.LOG_LEVEL}")
    print("=" * 50)

    # 检查模型状态
    from src.inference.model_manager import ModelManager
    manager = ModelManager()
    try:
        manager.load()
        print(f"\n[OK] 模型状态: 已加载 ({manager.n_labels} 个标签)")
        print(f"   标签: {manager.get_labels()}")
    except Exception as e:
        print(f"\n[ERR] 模型状态: 未加载 — {e}")

    print("\n运行 'python main.py --help' 查看更多选项\n")


# ============================================================
# 主入口
# ============================================================

def main(argv: Optional[list] = None) -> None:
    """主入口 — 解析命令行参数并分派子命令。"""
    parser = argparse.ArgumentParser(
        description="电商智能客服 NLU 意图识别引擎 — 企业级工程重构版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py train --model lr           训练逻辑回归
  python main.py train --all --compare       多模型对比
  python main.py train --model rf --grid-search  超参数调优
  python main.py serve                       启动推理服务
  python main.py predict --text "空调不制冷"  CLI 推理
  python main.py info                        系统信息
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="训练/评估模型")
    train_parser.add_argument(
        "--model", type=str, default="lr",
        choices=["lr", "rf", "svm", "all"],
        help="分类器类型: lr(逻辑回归) / rf(随机森林) / svm / all(全部对比)",
    )
    train_parser.add_argument(
        "--all", action="store_true",
        help="训练所有模型并选出最优",
    )
    train_parser.add_argument(
        "--compare", action="store_true",
        help="多模型对比实验",
    )
    train_parser.add_argument(
        "--grid-search", action="store_true",
        help="启用 GridSearchCV 超参数调优",
    )
    train_parser.add_argument(
        "--cv", type=int, default=3,
        help="交叉验证折数（默认 3）",
    )

    # ---- serve ----
    serve_parser = subparsers.add_parser("serve", help="启动 Flask 推理服务")
    serve_parser.add_argument("--host", type=str, default=None, help="绑定地址")
    serve_parser.add_argument("--port", type=int, default=None, help="绑定端口")
    serve_parser.add_argument("--debug", action="store_true", help="调试模式")

    # ---- predict ----
    predict_parser = subparsers.add_parser("predict", help="CLI 单次推理")
    predict_parser.add_argument(
        "--text", type=str, required=True, help="故障描述文本",
    )
    predict_parser.add_argument(
        "--threshold", type=float, default=settings.INFERENCE_THRESHOLD,
        help="置信度阈值 (0.0-1.0)",
    )
    predict_parser.add_argument(
        "--max-labels", type=int, default=settings.INFERENCE_MAX_LABELS,
        help="最大返回标签数",
    )

    # ---- info ----
    subparsers.add_parser("info", help="显示系统信息与配置")

    # 解析
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 确保输出目录存在
    settings.ensure_directories()

    # 分派命令
    command_map = {
        "train": _cmd_train,
        "serve": _cmd_serve,
        "predict": _cmd_predict,
        "info": _cmd_info,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
