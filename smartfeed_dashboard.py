import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFeed Dashboard",
    page_icon="🐄",
    layout="wide",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #212529; }
    .status-sehat  { background:#d4edda; color:#155724; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:600; }
    .status-berisiko { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:600; }
    .status-sakit  { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:600; }
    .section-title { font-size:18px; font-weight:600; margin: 1rem 0 0.5rem; color:#212529; }
</style>
""", unsafe_allow_html=True)

# ─── GENERATE DATASET & TRAIN MODELS (cached) ───────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    berat_badan        = np.random.normal(450, 50, n).clip(300, 600)
    porsi_konsentrat   = np.random.uniform(20, 60, n)
    porsi_hijauan      = 100 - porsi_konsentrat
    total_pakan_bk     = berat_badan * 0.033
    konsentrat_kg      = total_pakan_bk * (porsi_konsentrat / 100)
    hijauan_kg         = total_pakan_bk * (porsi_hijauan / 100)
    lama_laktasi       = np.random.randint(30, 270, n)
    umur_sapi          = np.random.randint(3, 9, n)

    produksi_susu = (
        -0.025 * (porsi_konsentrat - 38)**2
        + 0.04  * berat_badan
        - 0.03  * lama_laktasi
        + 5
        + np.random.normal(0, 1.5, n)
    ).clip(5, 35)

    ph_rumen = (
        7.0 - 0.035 * (porsi_konsentrat - 35).clip(0, None)
        + np.random.normal(0, 0.1, n)
    ).clip(5.5, 7.2)

    def assign_status(ph, susu, konsentrat):
        if ph < 6.0 or (konsentrat > 52 and ph < 6.3):
            return 'Sakit'
        elif ph < 6.4 or susu < 11:
            return 'Berisiko'
        else:
            return 'Sehat'

    status_kesehatan = [assign_status(ph_rumen[i], produksi_susu[i], porsi_konsentrat[i]) for i in range(n)]

    df = pd.DataFrame({
        'berat_badan_kg':       berat_badan.round(1),
        'porsi_konsentrat_pct': porsi_konsentrat.round(1),
        'porsi_hijauan_pct':    porsi_hijauan.round(1),
        'konsentrat_kg':        konsentrat_kg.round(2),
        'hijauan_kg':           hijauan_kg.round(2),
        'lama_laktasi_hari':    lama_laktasi,
        'umur_sapi_tahun':      umur_sapi,
        'ph_rumen':             ph_rumen.round(2),
        'produksi_susu_liter':  produksi_susu.round(2),
        'status_kesehatan':     status_kesehatan,
    })
    label_map = {'Sehat': 0, 'Berisiko': 1, 'Sakit': 2}
    df['status_kode'] = df['status_kesehatan'].map(label_map)
    return df

@st.cache_resource
def train_models(df):
    # --- Polynomial regression (degree 2) ---
    X_k = df[['porsi_konsentrat_pct']].values
    y_s = df['produksi_susu_liter'].values
    X_k_train, X_k_test, y_k_train, y_k_test = train_test_split(X_k, y_s, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    model_poly = LinearRegression()
    model_poly.fit(poly.fit_transform(X_k_train), y_k_train)

    # --- Decision Tree ---
    X_cls = df[['porsi_konsentrat_pct','konsentrat_kg','hijauan_kg','berat_badan_kg','ph_rumen','produksi_susu_liter']].values
    y_cls = df['status_kode'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
    dt.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, dt.predict(X_te))
    cm  = confusion_matrix(y_te, dt.predict(X_te))

    # --- KMeans ---
    X_cl = df[['porsi_konsentrat_pct','produksi_susu_liter','berat_badan_kg','ph_rumen']].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_cl)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = km.fit_predict(X_sc)

    return model_poly, poly, dt, acc, cm, df

df_raw   = load_data()
model_poly, poly, dt, acc, cm, df = train_models(df_raw)

COLORS   = {'Sehat':'#2ca02c', 'Berisiko':'#ff7f0e', 'Sakit':'#d62728'}
ORDER    = ['Sehat','Berisiko','Sakit']
FEAT_NAMES = ['porsi_konsentrat_%','konsentrat_kg','hijauan_kg','berat_badan_kg','ph_rumen','produksi_susu_liter']

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/id/thumb/6/6e/Institut_Pertanian_Bogor_logo.svg/240px-Institut_Pertanian_Bogor_logo.svg.png", width=80)
    st.markdown("### 🐄 SmartFeed")
    st.markdown("Sistem analitik kebutuhan pakan & kesehatan sapi perah berbasis data.")
    st.markdown("---")
    page = st.radio("Navigasi", ["📊 Ringkasan Data", "🔬 Hasil Model", "🧮 Kalkulator"])
    st.markdown("---")
    st.caption("Kelompok 3 – R4 | KOM 1231\nIPB University 2025/2026")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RINGKASAN DATA (EDA)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Ringkasan Data":
    st.title("📊 Ringkasan Data SmartFeed")
    st.caption("Dataset sintetis 200 observasi sapi perah FH berdasarkan standar NRC")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Data</div><div class="metric-value">{len(df)}</div></div>""", unsafe_allow_html=True)
    with c2:
        n_sehat = (df['status_kesehatan']=='Sehat').sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Sapi Sehat</div><div class="metric-value" style="color:#2ca02c">{n_sehat}</div></div>""", unsafe_allow_html=True)
    with c3:
        n_berisiko = (df['status_kesehatan']=='Berisiko').sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Berisiko</div><div class="metric-value" style="color:#ff7f0e">{n_berisiko}</div></div>""", unsafe_allow_html=True)
    with c4:
        n_sakit = (df['status_kesehatan']=='Sakit').sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Sapi Sakit</div><div class="metric-value" style="color:#d62728">{n_sakit}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Distribusi status + boxplot konsentrat
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Distribusi Status Kesehatan</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['status_kesehatan'].value_counts()
        bars = ax.bar(counts.index, counts.values, color=[COLORS[k] for k in counts.index], alpha=0.85, edgecolor='white', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Jumlah Sapi')
        ax.set_ylim(0, counts.max() + 15)
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-title">Porsi Konsentrat per Status</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        data_box = [df[df['status_kesehatan']==s]['porsi_konsentrat_pct'].values for s in ORDER]
        bp = ax.boxplot(data_box, labels=ORDER, patch_artist=True)
        for patch, s in zip(bp['boxes'], ORDER):
            patch.set_facecolor(COLORS[s]); patch.set_alpha(0.7)
        ax.axhline(38, color='blue', linestyle='--', linewidth=1.5, label='Optimal (38%)')
        ax.set_ylabel('Porsi Konsentrat (%)')
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Scatter + heatmap
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">Konsentrat vs Produksi Susu</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for s in ORDER:
            m = df['status_kesehatan'] == s
            ax.scatter(df.loc[m,'porsi_konsentrat_pct'], df.loc[m,'produksi_susu_liter'],
                       c=COLORS[s], label=s, alpha=0.6, s=30)
        xx = np.linspace(18, 62, 200).reshape(-1,1)
        ax.plot(xx, model_poly.predict(poly.transform(xx)), 'b-', linewidth=2, label='Kurva optimal')
        ax.axvline(38, color='orange', linestyle='--', linewidth=1.2)
        ax.set_xlabel('Porsi Konsentrat (%)'); ax.set_ylabel('Produksi Susu (liter/hari)')
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.markdown('<div class="section-title">Heatmap Korelasi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        corr_cols = ['berat_badan_kg','porsi_konsentrat_pct','konsentrat_kg','ph_rumen','produksi_susu_liter']
        corr = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax, linewidths=0.5, annot_kws={'size':9})
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tabel ringkasan
    st.markdown("---")
    st.markdown('<div class="section-title">Statistik Deskriptif</div>', unsafe_allow_html=True)
    cols_show = ['berat_badan_kg','porsi_konsentrat_pct','ph_rumen','produksi_susu_liter']
    st.dataframe(df[cols_show].describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HASIL MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Hasil Model":
    st.title("🔬 Hasil Pemodelan")

    tab1, tab2, tab3 = st.tabs(["📈 Regresi Polynomial", "🌳 Klasifikasi", "🔵 Clustering"])

    # ── TAB 1: REGRESI ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Hubungan konsentrat vs produksi susu bersifat non-linear (parabola).** Model polynomial degree 2 menangkap titik optimal di sekitar 38%.")
        col1, col2 = st.columns([2,1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.fill_between([35,41],[5,5],[35,35], alpha=0.10, color='green', label='Zona optimal (35–41%)')
            ax.fill_between([18,25],[5,5],[35,35], alpha=0.10, color='orange', label='Risiko ketosis (<25%)')
            ax.fill_between([50,62],[5,5],[35,35], alpha=0.10, color='red',    label='Risiko asidosis (>50%)')
            ax.scatter(df['porsi_konsentrat_pct'], df['produksi_susu_liter'], alpha=0.2, color='gray', s=20, label='Data')
            xx = np.linspace(18, 62, 300).reshape(-1,1)
            yy = model_poly.predict(poly.transform(xx))
            ax.plot(xx, yy, 'b-', linewidth=2.5, label='Model Poly Degree 2')
            ax.axvline(38, color='darkgreen', linestyle='--', linewidth=1.5)
            ax.text(38.5, 7, 'Optimal ~38%', color='darkgreen', fontsize=9)
            ax.set_xlabel('Porsi Konsentrat (% BK)'); ax.set_ylabel('Produksi Susu (liter/hari)')
            ax.legend(fontsize=8); sns.despine(ax=ax); fig.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            x_opt = xx[np.argmax(yy)][0]
            y_opt = yy.max()
            st.metric("Titik Optimal Konsentrat", f"{x_opt:.1f}%")
            st.metric("Prediksi Susu Maksimal",   f"{y_opt:.1f} liter/hari")
            st.info("Konsentrat < 25% → risiko **ketosis**\n\nKonsentrat > 50% → risiko **asidosis rumen**")

    # ── TAB 2: KLASIFIKASI ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(f"**Decision Tree** berhasil mengklasifikasikan status kesehatan dengan akurasi **{acc*100:.1f}%**.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Sehat','Berisiko','Sakit'],
                        yticklabels=['Sehat','Berisiko','Sakit'], linewidths=0.5)
            ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**Feature Importance**")
            importances = dt.feature_importances_
            sorted_idx  = np.argsort(importances)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            colors_fi = ['#d62728' if importances[i] > 0.3 else '#378ADD' if importances[i] > 0.1 else '#adb5bd' for i in sorted_idx]
            ax.barh([FEAT_NAMES[i] for i in sorted_idx], [importances[i] for i in sorted_idx], color=colors_fi, alpha=0.85)
            ax.set_xlabel('Importance')
            sns.despine(ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**Interpretasi variabel berpengaruh:**")
        st.markdown("""
| Variabel | Importance | Keterangan |
|---|---|---|
| `ph_rumen` | ⭐⭐⭐ Tertinggi | Indikator langsung asidosis rumen. pH < 6.0 = Sakit |
| `porsi_konsentrat_pct` | ⭐⭐⭐ Sangat tinggi | Rasio pakan menentukan keseimbangan rumen |
| `produksi_susu_liter` | ⭐⭐ Tinggi | Mencerminkan kondisi kesehatan secara tidak langsung |
| `berat_badan_kg` | ⭐ Sedang | Menentukan kebutuhan BK (~3.3% bobot badan) |
| `konsentrat_kg` & `hijauan_kg` | Rendah | Nilai absolut kurang representatif tanpa konteks berat |
""")

    # ── TAB 3: CLUSTERING ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("**K-Means (K=3)** mengelompokkan sapi ke 3 profil berbeda berdasarkan pola pakan dan produktivitas.")
        palette = ['#2ca02c','#ff7f0e','#d62728']
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            for c in range(3):
                m = df['cluster'] == c
                ax.scatter(df.loc[m,'porsi_konsentrat_pct'], df.loc[m,'produksi_susu_liter'],
                           c=palette[c], label=f'Cluster {c}', alpha=0.7, s=40)
            ax.axvline(38, color='black', linestyle='--', linewidth=1.2, label='Optimal (38%)')
            ax.set_xlabel('Porsi Konsentrat (%)'); ax.set_ylabel('Produksi Susu (liter/hari)')
            ax.legend(fontsize=9); sns.despine(ax=ax); fig.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            profil = df.groupby('cluster')[['porsi_konsentrat_pct','produksi_susu_liter','berat_badan_kg','ph_rumen']].mean().round(2)
            profil.index = [f'Cluster {i}' for i in profil.index]
            st.dataframe(profil, use_container_width=True)
            st.caption("Cluster dengan produksi susu tertinggi = pola pakan sudah optimal.\nCluster dengan pH rumen terendah = perlu intervensi pakan segera.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — KALKULATOR INTERAKTIF
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Kalkulator":
    st.title("🧮 Kalkulator Pakan & Status Kesehatan")
    st.markdown("Masukkan data sapi untuk mendapatkan rekomendasi pakan harian dan estimasi status kesehatan.")

    with st.form("kalkulator"):
        col1, col2 = st.columns(2)
        with col1:
            berat  = st.number_input("Berat badan sapi (kg)", min_value=200, max_value=700, value=450, step=10)
            pct_k  = st.slider("Porsi konsentrat (% dari total BK)", min_value=10, max_value=70, value=38)
            ph     = st.number_input("pH rumen (opsional, jika tersedia)", min_value=5.0, max_value=7.5, value=6.8, step=0.1)
        with col2:
            laktasi = st.number_input("Hari ke-berapa laktasi", min_value=1, max_value=305, value=90)
            umur    = st.number_input("Umur sapi (tahun)", min_value=1, max_value=15, value=4)
            produksi_aktual = st.number_input("Produksi susu aktual (liter/hari, opsional)", min_value=0.0, max_value=40.0, value=15.0, step=0.5)

        submitted = st.form_submit_button("🔍 Hitung", use_container_width=True)

    if submitted:
        # Hitung kebutuhan pakan
        total_bk      = berat * 0.033
        konsentrat_kg = total_bk * (pct_k / 100)
        hijauan_kg    = total_bk * (1 - pct_k / 100)
        pct_h         = 100 - pct_k

        # Prediksi produksi susu dari model polynomial
        pred_susu = model_poly.predict(poly.transform([[pct_k]]))[0]
        pred_susu = max(5, min(35, pred_susu))

        # Prediksi status kesehatan
        fitur = np.array([[pct_k, konsentrat_kg, hijauan_kg, berat, ph, produksi_aktual]])
        status_kode = dt.predict(fitur)[0]
        status_map  = {0:'Sehat', 1:'Berisiko', 2:'Sakit'}
        status      = status_map[status_kode]
        proba       = dt.predict_proba(fitur)[0]

        st.markdown("---")
        st.subheader("Hasil Kalkulasi")

        # Kebutuhan pakan
        c1, c2, c3 = st.columns(3)
        c1.metric("Total BK Pakan/hari", f"{total_bk:.2f} kg")
        c2.metric(f"Konsentrat ({pct_k}%)", f"{konsentrat_kg:.2f} kg")
        c3.metric(f"Hijauan ({pct_h}%)", f"{hijauan_kg:.2f} kg")

        st.markdown("---")

        # Status kesehatan
        col_s, col_r = st.columns([1, 2])
        with col_s:
            st.markdown("**Estimasi Status Kesehatan:**")
            badge_class = f"status-{status.lower()}"
            st.markdown(f'<span class="{badge_class}">{status}</span>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"Prediksi produksi susu: **{pred_susu:.1f} liter/hari**")

            # Probabilitas per kelas
            st.markdown("**Probabilitas:**")
            for label, p in zip(['Sehat','Berisiko','Sakit'], proba):
                st.progress(float(p), text=f"{label}: {p*100:.1f}%")

        with col_r:
            # Rekomendasi
            st.markdown("**Rekomendasi:**")
            if status == 'Sehat':
                st.success(f"""
✅ **Pakan sudah dalam kondisi optimal.**
- Porsi konsentrat {pct_k}% sudah {'ideal' if 35<=pct_k<=42 else 'dalam batas aman'}
- Pertahankan imbangan hijauan:konsentrat saat ini
- Lakukan pemantauan rutin setiap minggu
""")
            elif status == 'Berisiko':
                if pct_k > 42:
                    st.warning(f"""
⚠️ **Porsi konsentrat terlalu tinggi ({pct_k}%).**
- Kurangi konsentrat secara bertahap menuju 38–40%
- Tambah hijauan serat kasar untuk menjaga pH rumen
- Pantau pH rumen dan tanda-tanda asidosis subakut
""")
                else:
                    st.warning(f"""
⚠️ **Produksi susu di bawah potensi optimal.**
- Evaluasi kualitas hijauan yang diberikan
- Pertimbangkan suplementasi mineral dan vitamin
- Periksa kondisi kesehatan umum sapi
""")
            else:
                st.error(f"""
🚨 **Sapi membutuhkan perhatian segera.**
- pH rumen rendah → risiko asidosis rumen akut
- Hentikan sementara pemberian konsentrat tinggi
- Berikan hijauan/jerami kering untuk merangsang salivasi
- Konsultasikan dengan dokter hewan
""")

        # Visualisasi posisi konsentrat di kurva
        st.markdown("---")
        st.markdown("**Posisi porsi konsentrat di kurva optimal:**")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        xx  = np.linspace(18, 62, 300).reshape(-1,1)
        yy  = model_poly.predict(poly.transform(xx))
        ax.fill_between([35,41],[0,0],[35,35], alpha=0.12, color='green')
        ax.fill_between([18,25],[0,0],[35,35], alpha=0.10, color='orange')
        ax.fill_between([50,62],[0,0],[35,35], alpha=0.10, color='red')
        ax.plot(xx, yy, 'b-', linewidth=2.5)
        ax.axvline(pct_k, color='purple', linestyle='--', linewidth=2, label=f'Input kamu ({pct_k}%)')
        ax.scatter([pct_k], [pred_susu], color='purple', zorder=5, s=100)
        ax.text(pct_k + 0.5, pred_susu + 0.3, f'{pred_susu:.1f} L', color='purple', fontsize=10, fontweight='bold')
        ax.set_xlabel('Porsi Konsentrat (%)'); ax.set_ylabel('Prediksi Produksi Susu (liter/hari)')
        ax.legend(fontsize=9); sns.despine(ax=ax); fig.tight_layout()
        st.pyplot(fig); plt.close()

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SmartFeed · Kelompok 3 R4 · KOM 1231 Rekayasa Perangkat Lunak · IPB University 2025/2026")
