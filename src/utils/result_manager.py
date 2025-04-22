import os
import pandas as pd

def save_predictions_and_params(
    model_name: str,
    preds_df: pd.DataFrame,
    params: dict,
    result_base: str = 'results'
):
    """
    1) 모델별 서브폴더(results/<model_name>)를 만들고
    2) preds_df를 '<model_name>_<n>회차.csv'로 저장
    3) parameters.csv에 params + run 번호를 append
    """
    # 1. 경로 준비
    model_dir = os.path.join(result_base, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 2. run 카운트 계산
    existing = [
        f for f in os.listdir(model_dir)
        if f.startswith(f"{model_name}_") and f.endswith("회차.csv")
    ]
    run_no = len(existing) + 1

    # 3. 예측 저장
    pred_file = os.path.join(model_dir, f"{model_name}_{run_no}회차.csv")
    preds_df.to_csv(pred_file, index=False)
    print(f"✔ Predictions saved to {pred_file}")

    # 4. 파라미터 기록
    param_file = os.path.join(model_dir, "parameters.csv")
    entry = params.copy()
    entry.update({"run": run_no})
    df_new = pd.DataFrame([entry])

    if os.path.exists(param_file):
        df_old = pd.read_csv(param_file)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(param_file, index=False)
    print(f"✔ Parameters logged to {param_file}")