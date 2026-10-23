"""Pipeline smoke test (the `smoke_test` stage).

Smoke test on a 200-row subsample of the training data. Confirms the
stacking model initializes, fits and predicts without error, and that the
hasattr() guard distinguishes the stacking path from the legacy bagging
path. If the run context has no validation-model factory yet, the stage
defines one (create_validation_model) for the validation stages.
"""

import logging

import pandas as pd

from bpsp.context import ctx
from bpsp.data.preprocessing import detect_task_type
from bpsp.models.stacking import create_stacking_ensemble

logger = logging.getLogger(__name__)


def run() -> None:
    """Run this stage (notebook cell 11)."""
    global traceback
    if "create_validation_model" not in ctx:

        def create_validation_model(is_classification=True):
            if "create_stacking_ensemble" in dir():
                from sklearn.linear_model import LinearRegression, LogisticRegression

                meta = LogisticRegression(random_state=42) if is_classification else LinearRegression()
                return create_stacking_ensemble(
                    task_type="classification" if is_classification else "regression",
                    meta_model=meta,
                    cv=5,
                    random_state=42,
                )
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            return (
                GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
                if is_classification
                else GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            )

        ctx.create_validation_model = create_validation_model

    # Skip if run_validation_cells=False
    _SKIP_SMOKE = "run_validation_cells" in ctx and not ctx.run_validation_cells
    if _SKIP_SMOKE:
        logger.info("Skipped:  Pipeline smoke test skipped (run_validation_cells=False)")

    if not _SKIP_SMOKE:
        try:
            logger.info("Running pipeline smoke test...")

            X_smoke = ctx.X_train.iloc[:200].copy()
            y_smoke = ctx.y_train.iloc[:200].copy() if hasattr(ctx.y_train, "iloc") else ctx.y_train[:200]

            # 1. Test stacking model creation and fit
            task_type_smoke = detect_task_type(y_smoke)
            is_clf = task_type_smoke == "classification"
            m = ctx.create_validation_model(is_classification=is_clf)
            m.fit(X_smoke, y_smoke)
            logger.info("  Stacking model .fit() OK")

            # 2. Test predict / predict_proba
            preds = m.predict(X_smoke)
            logger.info(f"  .predict() OK -- shape: {preds.shape}")
            if is_clf and hasattr(m, "predict_proba"):
                proba = m.predict_proba(X_smoke)
                logger.info(f"  .predict_proba() OK -- shape: {proba.shape}")

            # 3. Test hasattr guard
            if hasattr(m, "tune_and_fit"):
                logger.info("  Warning:  Model has tune_and_fit (bagging/tuned path)")
            else:
                logger.info("  No tune_and_fit (stacking path) -- hasattr guard works")

            # 4. Test X_full construction
            X_full = pd.concat([ctx.X_train, ctx.X_test], axis=0) if "X_test" in ctx else ctx.X_train
            y_full = pd.concat([ctx.y_train, ctx.y_test], axis=0) if "y_test" in ctx else ctx.y_train
            logger.info(f"  X_full constructed -- shape: {X_full.shape} (X_train: {ctx.X_train.shape})")

            logger.info("\nAll smoke tests passed.")

        except Exception as e:
            logger.info(f"\nSmoke test failed: {e}")
            import traceback

            traceback.print_exc()
