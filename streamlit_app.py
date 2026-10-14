from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Scoring", page_icon="💳", layout="wide")
st.title("Credit Card Fraud Scoring")
st.caption("Research demo using the anonymized credit-card benchmark. Model scores are not payment decisions.")

MODEL_PATH=Path(__file__).parent/'artifacts'/'best_model.joblib'
if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run the notebook through the model-saving section first.")
    st.stop()
bundle=joblib.load(MODEL_PATH)
features=bundle['feature_names']; threshold=float(bundle.get('threshold',.5))

def score(frame):
    missing=[c for c in features if c not in frame.columns]
    extra=[c for c in frame.columns if c not in features]
    if missing: raise ValueError(f"Missing required feature columns: {missing}")
    if extra: st.warning(f"Ignoring extra columns: {extra}")
    x=frame[features].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    if x.isna().any().any(): raise ValueError("Inputs contain missing or non-finite values.")
    if bundle['model_type']=='sklearn':
        return bundle['model'].predict_proba(x)[:,1]
    import torch
    from torch import nn
    class FraudMLP(nn.Module):
        def __init__(self,n):
            super().__init__(); self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,1))
        def forward(self,x): return self.net(x).squeeze(1)
    model=FraudMLP(bundle['input_dim']); model.load_state_dict(bundle['state_dict']); model.eval()
    xx=bundle['scaler'].transform(x).astype('float32')
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(xx))).numpy()

st.write(f"Decision threshold from training validation: **{threshold:.4f}**")
st.write("Upload a CSV containing the 30 feature columns (`Time`, `V1`–`V28`, `Amount`).")
uploaded=st.file_uploader("Transaction feature CSV",type=['csv'])
if uploaded:
    try:
        frame=pd.read_csv(uploaded); probabilities=score(frame)
        out=frame.copy(); out['fraud_probability']=probabilities; out['predicted_fraud']=(probabilities>=threshold).astype('int8')
        st.dataframe(out.head(100),use_container_width=True)
        st.download_button("Download scored transactions",out.to_csv(index=False).encode(),"fraud_scores.csv","text/csv")
        st.metric("Transactions flagged",int(out['predicted_fraud'].sum()))
    except Exception as exc: st.error(str(exc))

with st.expander("Score one transaction"):
    st.caption("Enter the numeric benchmark features. For V1–V28, use the original PCA-transformed values.")
    with st.form('single'):
        values={}; cols=st.columns(3)
        for i,name in enumerate(features): values[name]=cols[i%3].number_input(name,value=0.0,format="%.6f")
        submitted=st.form_submit_button("Score transaction")
    if submitted:
        try:
            p=float(score(pd.DataFrame([values]))[0]); st.metric("Fraud probability",f"{p:.4%}")
            st.warning("Flag for review" if p>=threshold else "Below the selected review threshold")
        except Exception as exc: st.error(str(exc))

st.divider()
st.caption(f"Model: {bundle.get('model_name',bundle['model_type'])} | metrics: {bundle.get('metrics',{})}")
