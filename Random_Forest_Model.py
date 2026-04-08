import pandas as pd
import numpy as np
import urllib.parse
import html
import re
import joblib
from math import log2
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

def deep_decode(text):
    if not isinstance(text, str) or text == "":
        return "empty"
    try:
        for _ in range(2):
            text = urllib.parse.unquote(text)
            text = html.unescape(text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.lower()
    except:
        return str(text).lower()

def chunk_entropy(text, size=10):
    return max([
        calculate_entropy(text[i:i+size])
        for i in range(0, len(text), size)
    ]) if text else 0

def calculate_entropy(text):
    if not text:
        return 0
    counter = Counter(text)
    length = len(text)
    return -sum((count/length) * log2(count/length) for count in counter.values())

class StatisticalFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = []

        for text in posts:
            length = len(text)
            entropy = calculate_entropy(text)
            chunk_ent = chunk_entropy(text)

            special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
            digit_count = len(re.findall(r'\d', text))

            sqli_signs = text.count("'") + text.count("--") + text.count(";") + text.count("/*")
            xss_signs = text.count("<") + text.count(">") + text.count("script") + text.count("alert")
            path_signs = text.count("../") + text.count("etc/passwd")

            features.append([
                length,
                entropy,
                chunk_ent,
                special_chars / (length + 1),
                digit_count / (length + 1),
                sqli_signs,
                xss_signs,
                path_signs
            ])

        return np.array(features)

class AdvancedSecurityFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = []

        sql_keywords = [
            "select","union","insert","update","delete","drop",
            "where","or","and","sleep","benchmark",
            "from","not","like","in","exists"
        ]

        xss_keywords = [
            "<script>","<img","<iframe","javascript:",
            "alert","onerror","onload","eval"
        ]

        cmd_keywords = [
            "cmd","exec","system","bash","sh",
            "powershell","/bin/bash","/etc/passwd"
        ]

        # compile OUTSIDE loop
        sql_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sql_keywords)) + r')\b', re.IGNORECASE)
        xss_pattern = re.compile('|'.join(map(re.escape, xss_keywords)), re.IGNORECASE)
        cmd_pattern = re.compile('|'.join(map(re.escape, cmd_keywords)), re.IGNORECASE)

        for text in posts:

            length = len(text) + 1

            # ===== SQL =====
            sql_count = len(sql_pattern.findall(text))
            sql_ratio = sql_count / length

            # ===== XSS =====
            xss_count = len(xss_pattern.findall(text))
            xss_ratio = xss_count / length

            # ===== CMD =====
            cmd_count = len(cmd_pattern.findall(text))
            cmd_ratio = cmd_count / length

            # ===== Logic =====
            logic_true = int(bool(re.search(r'1\s*=\s*1|true', text, re.IGNORECASE)))
            logic_false = int(bool(re.search(r'1\s*=\s*0|false', text, re.IGNORECASE)))

            # ===== Obfuscation =====
            encoded = len(re.findall(r'%[0-9a-fA-F]{2}', text))
            hex_obfuscation = len(re.findall(r'\\x[0-9a-fA-F]{2}', text))
            unicode_obfuscation = len(re.findall(r'\\u[0-9a-fA-F]{4}', text))
            base64_like = len(re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', text))
            mixed_case = int(bool(re.search(r'[a-z][A-Z]|[A-Z][a-z]', text)))

            features.append([
                sql_count,
                sql_ratio,
                xss_count,
                xss_ratio,
                cmd_count,
                cmd_ratio,
                logic_true,
                logic_false,
                encoded,
                hex_obfuscation,
                unicode_obfuscation,
                base64_like,
                mixed_case
            ])

        return np.array(features)

class HeaderAnomalyFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = []

        for text in posts:

            user_agent_missing = int("user-agent" not in text)
            referer_missing = int("referer" not in text)

            suspicious_agents = int(any(bot in text for bot in [
                "sqlmap","nikto","crawler","bot","scan"
            ]))

            long_token = int(len(max(text.split(), key=len, default="")) > 50)

            many_params = int(text.count("=") > 5)

            unusual_method = int(any(m in text for m in [
                "put","delete","trace","connect"
            ]))

            features.append([
                user_agent_missing,
                referer_missing,
                suspicious_agents,
                long_token,
                many_params,
                unusual_method
            ])

        return np.array(features)

def print_academic_metrics(acc, precision, recall, f1, roc):
    html_output = f"""
    <div style="
        background:white;
        border:2px solid #1f4e79;
        padding:30px;
        border-radius:10px;
        width:650px;
        font-family:Segoe UI, Arial;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
    ">
        <h2 style="text-align:center;color:#1f4e79;">
            Model Performance Evaluation
        </h2>

        <table style="width:100%; font-size:18px; border-collapse:collapse;">
            <tr><td><b>Accuracy</b></td><td align="right">{acc:.4f}</td></tr>
            <tr style="background:#f2f6fa;"><td><b>Precision</b></td><td align="right">{precision:.4f}</td></tr>
            <tr><td><b>Recall</b></td><td align="right">{recall:.4f}</td></tr>
            <tr style="background:#f2f6fa;"><td><b>F1-score</b></td><td align="right">{f1:.4f}</td></tr>
            <tr><td><b>ROC-AUC</b></td><td align="right" style="color:#1f4e79;"><b>{roc:.4f}</b></td></tr>
        </table>
    </div>
    """
    display(HTML(html_output))


def export_html_report(acc, precision, recall, f1, roc):
    html_content = f"""
    <html>
    <head>
        <title>WAF Performance Report</title>
    </head>
    <body style="font-family:Arial;background:#f4f6f9;padding:40px;">
        <div style="background:white;padding:30px;border-radius:10px;width:700px;margin:auto;">
            <h1 style="color:#1f4e79;text-align:center;">Web Application Firewall Report</h1>
            <hr>
            <ul style="font-size:18px;">
                <li><b>Accuracy:</b> {acc:.4f}</li>
                <li><b>Precision:</b> {precision:.4f}</li>
                <li><b>Recall:</b> {recall:.4f}</li>
                <li><b>F1-score:</b> {f1:.4f}</li>
                <li><b>ROC-AUC:</b> {roc:.4f}</li>
            </ul>
        </div>
    </body>
    </html>
    """

    with open("waf_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Đã xuất file waf_report.html")

def plot_advanced_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = np.array([
        [
            f"TN\n{tn}\n({cm_percent[0,0]:.2f}%)",
            f"FP\n{fp}\n({cm_percent[0,1]:.2f}%)"
        ],
        [
            f"FN\n{fn}\n({cm_percent[1,0]:.2f}%)",
            f"TP\n{tp}\n({cm_percent[1,1]:.2f}%)"
        ]
    ])

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="YlGnBu",
        cbar=True,
        linewidths=1,
        linecolor="gray",
        xticklabels=["Predicted Normal", "Predicted Attack"],
        yticklabels=["Actual Normal", "Actual Attack"]
    )

    plt.title("Advanced Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print("\nAdditional Detection Metrics:")
    print(f" True Positive Rate (TPR): {tpr:.4f}")
    print(f" True Negative Rate (TNR): {tnr:.4f}")
    print(f" False Positive Rate (FPR): {fpr:.4f}")
    print(f" False Negative Rate (FNR): {fnr:.4f}")

def execute_pro_waf_system(csv_path):

    print("=== [1] DATA CLEANING ===")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df['URL'] = df['URL'].fillna('')
    df['content'] = df['content'].fillna('')

    df['processed_text'] = (df['URL'] + " " + df['content']).apply(deep_decode)

    df['classification'] = df['classification'].astype(str).str.lower()
    y = df['classification'].apply(lambda x: 0 if 'normal' in x or x == '0' else 1)

    df = df.drop_duplicates(subset=['processed_text'])

    X = df['processed_text'].values
    y = y.loc[df.index].values

    print(f"Dataset size: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n=== [DATA SPLIT ANALYSIS] ===")

    print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
    print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")

    train_attack_ratio = np.sum(y_train) / len(y_train)
    test_attack_ratio = np.sum(y_test) / len(y_test)

    print("\nClass Distribution:")
    print(f"Train - Normal: {(1-train_attack_ratio)*100:.2f}% | Attack: {train_attack_ratio*100:.2f}%")
    print(f"Test  - Normal: {(1-test_attack_ratio)*100:.2f}% | Attack: {test_attack_ratio*100:.2f}%")

    plt.figure(figsize=(6,4))
    labels = ['Normal', 'Attack']
    train_counts = [np.sum(y_train==0), np.sum(y_train==1)]
    test_counts = [np.sum(y_test==0), np.sum(y_test==1)]

    x = np.arange(len(labels))

    plt.bar(x - 0.2, train_counts, width=0.4, label='Train')
    plt.bar(x + 0.2, test_counts, width=0.4, label='Test')

    plt.xticks(x, labels)
    plt.title("Class Distribution (Train vs Test)")
    plt.legend()
    plt.show()

    features_union = FeatureUnion([
        ('manual_stats', StatisticalFeatures()),
        ('advanced_security', AdvancedSecurityFeatures()),
        ('header_anomaly', HeaderAnomalyFeatures()),
        ('tfidf_nlp', TfidfVectorizer(
            analyzer='char',
            ngram_range=(2,4),
            max_features=700,
            min_df=3,
            max_df=0.9
        ))
    ])

    model = Pipeline([
        ('features', features_union),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', RandomForestClassifier(
            n_estimators=500,
            max_depth= None,
            min_samples_leaf=3,
            class_weight= {0:1, 1:1},
            n_jobs=-1,
            random_state=42
        ))
    ])

    print("=== [2] TRAINING MODEL ===")
    model.fit(X_train, y_train)

    feature_names = []

    feature_names += [
        "length","entropy","chunk_entropy","special_ratio","digit_ratio",
        "sqli_signs","xss_signs","path_signs"
    ]

    feature_names += [
        "sql_count","sql_ratio",
        "xss_count","xss_ratio",
        "cmd_count","cmd_ratio",
        "logic_true","logic_false",
        "encoded","hex","unicode","base64","mixed_case"
    ]

    feature_names += [
        "ua_missing","ref_missing","bot","long_token","many_params","method"
    ]

    tfidf = dict(model.named_steps['features'].transformer_list)['tfidf_nlp']
    tfidf_features = tfidf.get_feature_names_out()
    feature_names += list(tfidf_features)

    importances = model.named_steps['classifier'].feature_importances_

    indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top Important Features")
    plt.show()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # ===== EXTRA ANALYSIS =====

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== CLASS IMBALANCE DETAIL ===")
    print(f"Total Normal: {np.sum(y==0)}")
    print(f"Total Attack: {np.sum(y==1)}")

    # ===== CUSTOM THRESHOLD =====
    threshold = 0.45
    y_pred_custom = (y_prob > threshold).astype(int)

    print("\n=== CUSTOM THRESHOLD ===")
    print(classification_report(y_test, y_pred_custom))

    # ===== TOP FEATURES =====
    top_features = [(feature_names[i], importances[i]) for i in indices]
    print("\n=== TOP FEATURES ===")
    for f, v in top_features:
        print(f"{f}: {v:.4f}")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print_academic_metrics(acc, precision, recall, f1, roc)
    export_html_report(acc, precision, recall, f1, roc)

    plot_advanced_confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    joblib.dump(model, "final_model.pkl")
    print("Model saved successfully!")

    dataset_stats = {
    "total_samples": len(X),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "train_attack_ratio": float(train_attack_ratio),
    "test_attack_ratio": float(test_attack_ratio)
    }

    return model, dataset_stats

if __name__ == "__main__":
    final_model, dataset_stats = execute_pro_waf_system("csic_database.csv")

    feature_config = {
        "tfidf": {
            "analyzer": "char",
            "ngram_range": (2,4),
            "max_features": 1500,
            "min_df": 3,
            "max_df": 0.9
        },
        "features_used": [
            "StatisticalFeatures",
            "AdvancedSecurityFeatures",
            "HeaderAnomalyFeatures",
            "TF-IDF"
        ]
    }

    print("\n=== DATASET STATS ===")
    for k, v in dataset_stats.items():
        print(f"{k}: {v}")

    print("\n=== FEATURE CONFIG ===")
    for k, v in feature_config.items():
        print(f"{k}: {v}")

    joblib.dump(dataset_stats, "dataset_stats.pkl")
    joblib.dump(feature_config, "feature_config.pkl")

    print("\nĐã lưu dataset_stats.pkl và feature_config.pkl")