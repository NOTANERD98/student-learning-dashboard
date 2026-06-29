"""
Streamlit Dashboard – Student Learning Patterns & Dropout Risk System
Run: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Matplotlib font fallback (cross-platform Hebrew glyphs) ───────────────────
# Linux hosts (e.g. Streamlit Community Cloud) ship without Hebrew-capable fonts,
# so chart labels can render as empty boxes. Provide a fallback chain so Hebrew
# glyphs in figures use an available font (install fonts-noto-core via
# packages.txt). Harmless elsewhere — matplotlib silently skips missing fonts.
plt.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans Hebrew', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="מערכת זיהוי דפוסי למידה",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (RTL) ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { direction: rtl; text-align: right; }
    .stApp { direction: rtl; }
    .main-title { font-size: 2.2rem; font-weight: bold; color: #2c3e50; text-align: center; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 15px;
                   border-left: 4px solid #3498db; margin: 5px 0; }
    .success-box { background: #d5f5e3; border-radius: 8px; padding: 10px; }
    .warning-box { background: #fadbd8; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Data loading / generation ────────────────────────────────────────────────
@st.cache_data
def load_or_generate_data():
    csv_path = "data/students_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        N = 1000
        prev = np.random.normal(72, 15, N).clip(40, 100)
        quiz = (prev * 0.55 + np.random.normal(0, 13, N)).clip(0, 100)
        logins = np.random.normal(45, 22, N).clip(1, 150).astype(int)
        sess  = np.random.normal(35, 15, N).clip(5, 120)
        assgn = np.random.normal(11, 3, N).clip(0, 15).astype(int)
        late  = np.random.normal(2, 1.8, N).clip(0, 10).astype(int)
        forum = np.random.normal(8, 6, N).clip(0, 50).astype(int)
        vrate = np.random.normal(0.68, 0.22, N).clip(0, 1)
        final = (prev*0.25 + quiz*0.35 + (assgn/15)*18 +
                 vrate*12 - (late/10)*10 + (logins/150)*6 +
                 np.random.normal(0, 7, N)).clip(0, 100)
        df = pd.DataFrame({
            'student_id': range(1001, 1001+N),
            'number_of_logins': logins,
            'avg_session_time': sess.round(1),
            'assignments_submitted': assgn,
            'late_submissions': late,
            'quiz_avg': quiz.round(1),
            'forum_posts': forum,
            'video_completion_rate': vrate.round(3),
            'previous_grade': prev.round(1),
            'final_grade': final.round(1),
            'success': (final >= 70).astype(int)
        })
        np.random.seed(123)
        for col in ['avg_session_time', 'quiz_avg', 'forum_posts', 'video_completion_rate']:
            idx = np.random.choice(N, int(N*0.05), replace=False)
            df.loc[idx, col] = np.nan
    return df

@st.cache_resource
def train_models(df):
    FEAT = ['number_of_logins','avg_session_time','assignments_submitted',
            'late_submissions','quiz_avg','forum_posts',
            'video_completion_rate','previous_grade']
    X = df[FEAT].copy()
    y = df['success']

    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEAT)

    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2,
                                                random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc  = sc.transform(X_te)

    models = {
        "Decision Tree"      : DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        "KNN (K=9)"          : KNeighborsClassifier(n_neighbors=9),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Naive Bayes"        : GaussianNB(),
        "Neural Network"     : MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300,
                                             early_stopping=True, random_state=42),
        "Gradient Boosting ★": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                           learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(Xtr_sc, y_tr)
        yp   = m.predict(Xte_sc)
        yprb = m.predict_proba(Xte_sc)[:,1] if hasattr(m, 'predict_proba') else None
        results[name] = {
            'model'    : m,
            'Accuracy' : accuracy_score(y_te, yp),
            'Precision': precision_score(y_te, yp, zero_division=0),
            'Recall'   : recall_score(y_te, yp, zero_division=0),
            'F1'       : f1_score(y_te, yp, zero_division=0),
            'AUC'      : roc_auc_score(y_te, yprb) if yprb is not None else None,
            'cm'       : confusion_matrix(y_te, yp),
            'y_pred'   : yp,
            'y_proba'  : yprb,
            'y_test'   : y_te.values,
        }
    return results, Xte_sc, y_te, FEAT, sc, imp

# ─── Load ──────────────────────────────────────────────────────────────────────
df = load_or_generate_data()
results, Xte_sc, y_te, FEAT, sc, imp = train_models(df)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎓 ניווט")
page = st.sidebar.radio("בחר מסך:", [
    "🏠 סקירה כללית",
    "📊 ניתוח נתונים (EDA)",
    "🤖 השוואת מודלים",
    "🔍 חיזוי סטודנט",
    "🗂️ אשכולות (Clustering)",
    "⚖️ ניתוח הוגנות"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**פרויקט:** כריית נתונים  \n**מוסד:** כנרת  \n**סמסטר:** ב׳ תשפ״ו")

# ─── Page 1: Overview ─────────────────────────────────────────────────────────
if page == "🏠 סקירה כללית":
    st.markdown('<p class="main-title">🎓 מערכת לזיהוי דפוסי למידה וסיכון לנשירה</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("סה״כ סטודנטים", f"{len(df):,}")
    c2.metric("שיעור הצלחה", f"{df['success'].mean():.1%}")
    c3.metric("ממוצע ציון סופי", f"{df['final_grade'].mean():.1f}")
    c4.metric("מאפיינים", len(FEAT))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 שאלת המחקר")
        st.info("""
**האם ניתן לחזות הצלחה בקורס על סמך:**
- דפוסי למידה (כניסות, זמן שהייה)
- השתתפות (פורום, מטלות)
- ציוני ביניים
- התנהגות במערכת למידה מקוונת
        """)
        st.subheader("🎯 מטרות הפרויקט")
        st.markdown("""
1. בניית מודל חיזוי הצלחה/כישלון
2. זיהוי קבוצות סטודנטים לפי דפוסי למידה
3. השוואת ביצועי מודלים שונים
4. פרשנות והמלצות יישומיות
        """)

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        df['success'].value_counts().plot(kind='pie', ax=axes[0],
            labels=['הצלחה', 'כישלון'], colors=['#2ecc71','#e74c3c'],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        axes[0].set_title('התפלגות הצלחה/כישלון', fontweight='bold')
        axes[0].set_ylabel('')

        df['final_grade'].hist(bins=30, ax=axes[1], color='#3498db',
                               edgecolor='white', alpha=0.85)
        axes[1].axvline(70, color='red', linestyle='--', linewidth=2, label='Cutoff=70')
        axes[1].set_title('התפלגות ציון סופי', fontweight='bold')
        axes[1].set_xlabel('ציון'); axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📋 תיאור המשתנים")
    var_desc = pd.DataFrame({
        'משתנה': FEAT + ['final_grade', 'success'],
        'סוג': ['מספרי']*len(FEAT) + ['מספרי', 'בינארי'],
        'תיאור': [
            'מספר כניסות למערכת הלמידה',
            'זמן ממוצע (דקות) בכל כניסה',
            'מספר מטלות שהוגשו (מתוך 15)',
            'מספר הגשות שאוחרו',
            'ממוצע ציוני בחנים (0-100)',
            'מספר הודעות שנכתבו בפורום',
            'אחוז צפייה בסרטוני הקורס (0-1)',
            'ציון קודם (מקורס קודם, 0-100)',
            'ציון סופי בקורס (0-100)',
            'הצלחה: 1 אם ציון≥70, אחרת 0'
        ]
    })
    st.dataframe(var_desc, hide_index=True, use_container_width=True)

# ─── Page 2: EDA ──────────────────────────────────────────────────────────────
elif page == "📊 ניתוח נתונים (EDA)":
    st.title("📊 ניתוח נתונים ראשוני (EDA)")

    tab1, tab2, tab3 = st.tabs(["📈 התפלגויות", "🔗 קורלציות", "📦 Boxplots"])

    with tab1:
        feat_sel = st.selectbox("בחר משתנה", FEAT + ['final_grade'])
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            data = df[feat_sel].dropna()
            ax.hist(data, bins=35, color='#3498db', edgecolor='white', alpha=0.85)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={data.mean():.1f}')
            ax.axvline(data.median(), color='orange', linestyle=':', linewidth=2,
                       label=f'Median={data.median():.1f}')
            ax.set_title(f'התפלגות: {feat_sel}', fontweight='bold')
            ax.legend()
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**סטטיסטיקות תיאוריות**")
            stats = df[feat_sel].describe().round(3)
            st.dataframe(stats.to_frame('ערך'))
            miss = df[feat_sel].isnull().sum()
            if miss > 0:
                st.warning(f"⚠️ {miss} ערכים חסרים ({miss/len(df)*100:.1f}%)")

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.drop('student_id', axis=1).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                    mask=mask, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=0.5, ax=ax, annot_kws={'size': 9})
        ax.set_title('מטריצת קורלציות', fontsize=14, fontweight='bold')
        st.pyplot(fig); plt.close()

        st.markdown("**קורלציה עם SUCCESS:**")
        corr_s = df.drop(['student_id','success'], axis=1).corrwith(df['success']).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in corr_s]
        ax2.barh(corr_s.index, corr_s.values, color=colors, edgecolor='white', alpha=0.9)
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_title('Correlation with Success', fontweight='bold')
        st.pyplot(fig2); plt.close()

    with tab3:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        palette = {0: '#e74c3c', 1: '#2ecc71'}
        for i, feat in enumerate(FEAT):
            ax = axes[i // 4][i % 4]
            df_c = df[['success', feat]].dropna()
            ax.boxplot([df_c[df_c['success']==0][feat],
                        df_c[df_c['success']==1][feat]],
                       labels=['כישלון', 'הצלחה'],
                       patch_artist=True,
                       boxprops=dict(alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
            boxes = ax.findobj(plt.matplotlib.patches.PathPatch)
            if len(boxes) >= 2:
                boxes[0].set_facecolor('#e74c3c')
                boxes[1].set_facecolor('#2ecc71')
            ax.set_title(feat.replace('_',' ').title(), fontsize=8, fontweight='bold')
        plt.suptitle('Boxplots: כישלון (אדום) vs הצלחה (ירוק)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ─── Page 3: Model Comparison ─────────────────────────────────────────────────
elif page == "🤖 השוואת מודלים":
    st.title("🤖 השוואת מודלים – למידה מונחית")

    # Table
    rows = [{k: v for k, v in r.items() if k not in ['model','cm','y_pred','y_proba','y_test']}
            for r in results.values()]
    comp_df = pd.DataFrame(rows, index=results.keys()).sort_values('F1', ascending=False)
    comp_df['AUC'] = comp_df['AUC'].map(lambda x: f'{x:.4f}' if x else 'N/A')

    st.subheader("📊 טבלת השוואה")
    st.dataframe(comp_df.style
                 .highlight_max(subset=['Accuracy','Precision','Recall','F1'], color='#d5f5e3')
                 .highlight_min(subset=['Accuracy','Precision','Recall','F1'], color='#fadbd8')
                 .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}','F1':'{:.4f}'}),
                 use_container_width=True)

    st.markdown("---")
    selected_model = st.selectbox("בחר מודל לפרטים:", list(results.keys()))
    r = results[selected_model]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['כישלון','הצלחה'], yticklabels=['כישלון','הצלחה'],
                    cbar=False, annot_kws={'size': 16})
        ax.set_xlabel('חיזוי'); ax.set_ylabel('אמיתי')
        ax.set_title(selected_model, fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("ROC Curve")
        if r['y_proba'] is not None:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_proba'])
            ax2.plot(fpr, tpr, color='#3498db', linewidth=2.5,
                     label=f"AUC = {r['AUC']:.3f}")
            ax2.plot([0,1],[0,1], 'k--', linewidth=1)
            ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve', fontweight='bold')
            ax2.legend(); ax2.grid(alpha=0.3)
            st.pyplot(fig2); plt.close()
        else:
            st.info("ROC לא זמין למודל זה")

    # Feature importance
    if hasattr(r['model'], 'feature_importances_'):
        st.subheader("Feature Importance")
        fi = pd.Series(r['model'].feature_importances_, index=FEAT).sort_values(ascending=True)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        fi.plot(kind='barh', color='#3498db', edgecolor='white', ax=ax3)
        ax3.set_title(f'Feature Importance – {selected_model}', fontweight='bold')
        st.pyplot(fig3); plt.close()

# ─── Page 4: Predict ──────────────────────────────────────────────────────────
elif page == "🔍 חיזוי סטודנט":
    st.title("🔍 חיזוי הצלחה לסטודנט חדש")

    st.info("הזן פרמטרים עבור סטודנט ובחר מודל לקבלת חיזוי")

    col1, col2 = st.columns(2)
    with col1:
        logins  = st.slider("מספר כניסות", 1, 150, 45)
        sess_t  = st.slider("זמן ממוצע (דק')", 5.0, 120.0, 35.0)
        assgn   = st.slider("מטלות שהוגשו", 0, 15, 11)
        late    = st.slider("הגשות באיחור", 0, 10, 2)

    with col2:
        quiz    = st.slider("ממוצע בחנים", 0.0, 100.0, 70.0)
        forum   = st.slider("הודעות בפורום", 0, 50, 8)
        video   = st.slider("אחוז צפייה בסרטונים", 0.0, 1.0, 0.7, 0.05)
        prev_g  = st.slider("ציון קודם", 40.0, 100.0, 72.0)

    model_choice = st.selectbox("בחר מודל:", list(results.keys()))
    predict_btn  = st.button("🔮 חזה!")

    if predict_btn:
        inp = np.array([[logins, sess_t, assgn, late, quiz, forum, video, prev_g]])
        inp_scaled = sc.transform(imp.transform(inp))
        model_obj  = results[model_choice]['model']
        pred = model_obj.predict(inp_scaled)[0]
        prob = model_obj.predict_proba(inp_scaled)[0][1] if hasattr(model_obj, 'predict_proba') else None

        st.markdown("---")
        if pred == 1:
            st.success(f"✅ **הסטודנט צפוי להצליח!**" +
                       (f"  |  הסתברות: **{prob:.1%}**" if prob else ""))
        else:
            st.error(f"⚠️ **הסטודנט בסיכון לקושי / כישלון**" +
                     (f"  |  הסתברות הצלחה: **{prob:.1%}**" if prob else ""))

        if prob is not None:
            st.progress(float(prob))

        # What-if analysis
        st.subheader("💡 ניתוח 'מה-אם' – שיפור פוטנציאלי")
        suggestions = []
        if late > 3:
            suggestions.append(f"הפחת הגשות באיחור מ-{late} ל-1 → שיפור צפוי בציון")
        if video < 0.6:
            suggestions.append(f"הגדל צפייה בסרטונים מ-{video:.0%} ל-80% → מעורבות גבוהה יותר")
        if assgn < 10:
            suggestions.append(f"הגש יותר מטלות: {assgn}/15 → כוון ל-13/15")
        if quiz < 60:
            suggestions.append(f"ממוצע בחנים נמוך ({quiz:.0f}) → נדרש שיפור מהותי")
        if forum == 0:
            suggestions.append("לא פעיל בפורום → השתתפות בדיונים מועילה")

        if suggestions:
            for s in suggestions:
                st.warning(f"• {s}")
        else:
            st.success("• הסטודנט מציג ביצועים טובים בכל המדדים!")

# ─── Page 5: Clustering ───────────────────────────────────────────────────────
elif page == "🗂️ אשכולות (Clustering)":
    st.title("🗂️ ניתוח אשכולות – זיהוי דפוסי למידה")

    @st.cache_resource
    def run_clustering(df):
        FEAT = ['number_of_logins','avg_session_time','assignments_submitted',
                'late_submissions','quiz_avg','forum_posts',
                'video_completion_rate','previous_grade']
        X = df[FEAT].copy()
        imp2 = SimpleImputer(strategy='median')
        X_imp = imp2.fit_transform(X)
        sc2 = StandardScaler()
        X_sc = sc2.fit_transform(X_imp)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_sc)
        return X_sc, X_pca, pca, FEAT

    X_sc_cl, X_pca_cl, pca_cl, FEAT_cl = run_clustering(df)

    k = st.slider("מספר אשכולות (K)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc_cl)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca_cl[:,0], X_pca_cl[:,1],
                             c=labels, cmap='tab10', alpha=0.6, s=15)
        centers_pca = pca_cl.transform(km.cluster_centers_)
        ax.scatter(centers_pca[:,0], centers_pca[:,1],
                   c='black', marker='X', s=200, zorder=5, label='Centroids')
        ax.set_title(f'K-Means (K={k}) – מרחב PCA', fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca_cl.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_cl.explained_variance_ratio_[1]:.1%})')
        ax.legend()
        plt.colorbar(scatter)
        st.pyplot(fig); plt.close()

    with col2:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X_sc_cl, labels)
        st.metric("Silhouette Score", f"{sil:.4f}",
                  help="גבוה יותר = אשכולות מוגדרים יותר (מקסימום 1.0)")

        vc = pd.Series(labels).value_counts().sort_index()
        st.markdown("**גודל כל אשכול:**")
        for cluster_id, count in vc.items():
            st.write(f"  Cluster {cluster_id}: {count} סטודנטים")

        # Cluster success rate
        df_cl = df.copy()
        df_cl['cluster'] = labels
        st.markdown("**שיעור הצלחה לפי אשכול:**")
        for cid in sorted(df_cl['cluster'].unique()):
            rate = df_cl[df_cl['cluster'] == cid]['success'].mean()
            emoji = "✅" if rate > 0.7 else ("⚠️" if rate > 0.4 else "❌")
            st.write(f"  {emoji} Cluster {cid}: {rate:.1%}")

# ─── Page 6: Fairness ─────────────────────────────────────────────────────────
elif page == "⚖️ ניתוח הוגנות":
    st.title("⚖️ ניתוח הוגנות והטיות (Fairness & Bias)")

    st.info("""
**שאלות לניתוח:**
- האם המודל מבצע באופן שווה עבור קבוצות שונות?
- האם ישנה הטיה לטובת/נגד קבוצת סטודנטים מסוימת?
    """)

    # Group by prior grade
    # IMPORTANT FIX:
    # Xte_sc / y_te contain only the test set (200 rows), while df contains all students (1000 rows).
    # Therefore, build df_f only from the same test rows before adding predictions.
    df_f = df.loc[y_te.index].copy()
    df_f['y_true'] = y_te.values
    df_f['y_pred'] = results['Random Forest']['y_pred']
    df_f['perf_group'] = pd.qcut(df_f['previous_grade'], q=3, labels=['נמוך','בינוני','גבוה'])

    fairness_rows = []
    for grp in ['נמוך', 'בינוני', 'גבוה']:
        m = df_f['perf_group'] == grp
        if m.sum() < 5: continue
        yt = df_f.loc[m, 'y_true']
        yp = df_f.loc[m, 'y_pred']
        fairness_rows.append({
            'קבוצה': grp, 'N': m.sum(),
            'Accuracy': accuracy_score(yt, yp),
            'Precision': precision_score(yt, yp, zero_division=0),
            'Recall': recall_score(yt, yp, zero_division=0),
            'F1': f1_score(yt, yp, zero_division=0)
        })

    if fairness_rows:
        fair_df = pd.DataFrame(fairness_rows)
        st.subheader("ביצועי המודל לפי קבוצת ציון קודם")
        st.dataframe(fair_df.set_index('קבוצה').round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(fair_df))
        w = 0.2
        for i, m in enumerate(['Accuracy','Precision','Recall','F1']):
            ax.bar(x + i*w, fair_df[m], w, label=m, alpha=0.85, edgecolor='white')
        ax.set_xticks(x + 1.5*w); ax.set_xticklabels(fair_df['קבוצה'])
        ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
        ax.set_title('ביצועי Random Forest לפי קבוצה', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.subheader("📋 פרשנות והמלצות")
        st.warning("""
**ממצאים:**
- המודל משיג ביצועים טובים יותר לקבוצת הציון הגבוה
- קבוצת הציון הנמוך מציגה Recall נמוך יותר → המודל מחמיץ סטודנטים בסיכון

**המלצות:**
1. שקול שימוש בסף (threshold) שונה לכל קבוצה
2. הוסף נתוני רקע סוציו-אקונומי לניתוח הטיות עמוק יותר
3. הכשר מחדש עם דגש על הוגנות (Fairness-aware training)
4. בדוק עיוות בנתוני האימון (class imbalance within groups)
        """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("🤖 Claude AI + ChatGPT were used to develop this project")
