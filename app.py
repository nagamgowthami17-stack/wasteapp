
import streamlit as st
import torch
import pandas as pd
from PIL import Image
from datetime import datetime
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="GreenVision | AI Waste Management",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_NAME = "watersplash/waste-classification"


# =====================================================
# CREATIVE CSS DESIGN
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #f3fff7 0%, #f3f8ff 55%, #f9f5ff 100%);
    color: #20372c;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Readable general text */
p, li, label {
    color: #263e32;
}

h1, h2, h3, h4 {
    color: #145c3c !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e4fff0, #e8f3ff);
    border-right: 1px solid #c8e8d5;
}

[data-testid="stSidebar"] * {
    color: #17442e;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #ffffff;
    color: #12633f !important;
    border: 1px solid #c5e8d1;
    border-radius: 12px;
    font-weight: 700;
    min-height: 43px;
    text-align: left;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #d7f9e5;
    border: 1px solid #15945c;
    color: #075b35 !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #064e3b, #059669, #0891b2);
    padding: 48px 25px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 10px 28px rgba(0, 100, 70, 0.16);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 48px;
    font-weight: 850;
    margin-bottom: 8px;
}

.hero p {
    color: #ffffff !important;
    font-size: 17px;
}

/* Home page mini banner */
.welcome-banner {
    background: linear-gradient(110deg, #dcfce7, #dbeafe);
    border: 1px solid #b9e7cd;
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0 25px 0;
}

.welcome-banner h2 {
    color: #075b3a !important;
}

.welcome-banner p {
    color: #315746 !important;
    font-size: 16px;
}

/* Section headings */
.section-title {
    color: #087c55 !important;
    font-size: 27px;
    font-weight: 800;
    margin: 20px 0;
}

/* Feature cards */
.feature-card {
    background: #ffffff;
    border: 1px solid #d5eadd;
    border-radius: 18px;
    padding: 22px;
    min-height: 150px;
    box-shadow: 0 5px 18px rgba(25, 90, 55, 0.06);
    margin-bottom: 15px;
}

.feature-card h3 {
    color: #087c55 !important;
    font-size: 19px;
}

.feature-card p {
    color: #435b4d !important;
    font-size: 14px;
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, #ffffff, #e8fff0);
    border: 1px solid #bde8cd;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 5px 16px rgba(20, 100, 60, 0.07);
    margin: 10px 0;
}

.result-label {
    color: #547563 !important;
    font-size: 14px;
    font-weight: 700;
}

.result-value {
    color: #087c4b !important;
    font-size: 29px;
    font-weight: 850;
}

/* Disposal recommendation */
.tip-card {
    background: linear-gradient(110deg, #dcfce7, #e0f2fe);
    border-left: 6px solid #059669;
    border-radius: 14px;
    padding: 22px;
    color: #174b35 !important;
    font-weight: 600;
    margin: 15px 0;
}

/* Category cards */
.category-card {
    background: #ffffff;
    border: 1px solid #d4e9dc;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(20, 80, 50, 0.05);
}

.category-card h4 {
    color: #087c55 !important;
}

.category-card p {
    color: #385548 !important;
}

/* Main buttons */
.stButton > button {
    background: linear-gradient(90deg, #07864f, #08a89a);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    font-weight: 750;
    min-height: 45px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #056b40, #078c83);
    color: #ffffff !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(90deg, #1769aa, #248ed0);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    min-height: 43px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #cce9d7;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(20, 80, 50, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #426d55 !important;
}

[data-testid="stMetricValue"] {
    color: #087c55 !important;
}

/* Image uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #62b98a;
    border-radius: 16px;
    padding: 18px;
}

/* Progress */
.stProgress > div > div > div {
    background: #00a878;
}

/* Footer */
.footer {
    text-align: center;
    color: #47735b;
    font-weight: 700;
    padding: 22px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# AI MODEL
# =====================================================

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


# =====================================================
# WASTE CATEGORIES
# =====================================================

disposal_tips = {
    "Battery": "🔋 Take batteries to an authorised battery recycling collection point.",
    "Biological": "🌱 Place suitable organic waste in a composting bin.",
    "Brown Glass": "🍾 Clean and place in a glass recycling collection bin.",
    "Cardboard": "📦 Keep dry, flatten, and place in paper recycling.",
    "Clothes": "👕 Donate wearable clothes or use textile recycling.",
    "Green Glass": "🍾 Clean and place in a glass recycling collection bin.",
    "Metal": "🥫 Separate metal items for metal recycling.",
    "Paper": "📄 Keep paper dry and place it in a paper recycling bin.",
    "Plastic": "🧴 Check the recycling symbol and local recycling rules.",
    "Shoes": "👟 Donate usable shoes or look for footwear recycling.",
    "Trash": "🗑️ Place in the appropriate general waste bin.",
    "White Glass": "🍾 Clean and place in a glass recycling collection bin."
}

category_descriptions = {
    "Battery": "Used batteries and battery-related waste.",
    "Biological": "Organic waste such as food scraps and plant material.",
    "Brown Glass": "Brown-coloured glass bottles and containers.",
    "Cardboard": "Boxes, packaging and cardboard sheets.",
    "Clothes": "Used clothing and textile materials.",
    "Green Glass": "Green-coloured glass bottles and containers.",
    "Metal": "Metal cans and other recyclable metal items.",
    "Paper": "Newspapers, sheets and paper materials.",
    "Plastic": "Plastic bottles, containers and packaging.",
    "Shoes": "Footwear and used shoes.",
    "Trash": "General waste that does not fit the other categories.",
    "White Glass": "Clear or white glass bottles and containers."
}

category_emojis = {
    "Battery": "🔋",
    "Biological": "🌱",
    "Brown Glass": "🍾",
    "Cardboard": "📦",
    "Clothes": "👕",
    "Green Glass": "🟢",
    "Metal": "🥫",
    "Paper": "📄",
    "Plastic": "🧴",
    "Shoes": "👟",
    "Trash": "🗑️",
    "White Glass": "🥛"
}


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


def go_to_page(page):
    st.session_state.current_page = page


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#087c55;font-size:29px;'>🌿 GreenVision</h1>",
        unsafe_allow_html=True
    )

    st.caption("SMART WASTE MANAGEMENT SYSTEM")

    st.divider()

    st.subheader("🧭 Main Menu")

    navigation = [
        ("🏠 Home", "Home"),
        ("🤖 AI Image Classification", "Classifier"),
        ("♻️ 12 Waste Categories", "Categories"),
        ("📈 Admin Dashboard", "Dashboard"),
        ("📚 Classification History", "History"),
        ("📥 CSV Reports", "CSV Report")
    ]

    for label, page_name in navigation:
        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True
        ):
            go_to_page(page_name)

    st.divider()

    st.subheader("📊 Quick Statistics")

    st.metric(
        "Total Images Classified",
        len(st.session_state.history)
    )

    if st.button(
        "🗑️ Clear All History",
        key="clear_history",
        use_container_width=True
    ):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    st.caption("🌍 Reduce • Reuse • Recycle")


# =====================================================
# COMMON HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🌿 GreenVision</h1>
    <p>SMART WASTE CLASSIFICATION SYSTEM</p>
    <p>♻️ Intelligent Waste Management | AI Powered | Eco Friendly</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# HOME PAGE
# =====================================================

if st.session_state.current_page == "Home":

    st.markdown("""
    <div class="welcome-banner">
        <h2>🌎 Welcome to GreenVision!</h2>
        <p>
        A smarter way to identify and manage waste.
        GreenVision uses Artificial Intelligence to classify waste images
        and provide useful disposal recommendations.
        Let's make our environment cleaner and greener!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">✨ Explore GreenVision</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Image Classification</h3>
            <p>Upload a waste image and let our AI model predict its category.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>♻️ 12 Waste Categories</h3>
            <p>Explore the different types of waste supported by our AI model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🌱 Disposal Recommendations</h3>
            <p>Get useful guidance on how to handle different waste categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Admin Dashboard</h3>
            <p>View classification statistics and category-wise charts.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>📚 Classification History</h3>
            <p>Review images classified during your current app session.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>📥 CSV Reports</h3>
            <p>Download classification records and category summaries.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚀 Get Started")

    if st.button(
        "🤖 Start Waste Classification",
        use_container_width=True
    ):
        go_to_page("Classifier")

    st.divider()

    st.markdown("### 📊 Project Overview")

    total = len(st.session_state.history)

    col1, col2, col3 = st.columns(3)

    col1.metric("📸 Images Classified", total)
    col2.metric("♻️ Supported Categories", 12)
    col3.metric("🌿 Project", "GreenVision")


# =====================================================
# AI CLASSIFICATION PAGE
# =====================================================

elif st.session_state.current_page == "Classifier":

    st.markdown(
        '<div class="section-title">🤖 AI Waste Image Classification</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a clear waste image to identify its category."
    )

    uploaded_file = st.file_uploader(
        "📸 Choose your waste image",
        type=["jpg", "jpeg", "png"],
        key="waste_image"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Waste Image",
            width=400
        )

        if st.button(
            "🔍 ANALYZE WASTE IMAGE",
            use_container_width=True
        ):

            try:
                with st.spinner("🌿 GreenVision AI is analyzing your image..."):

                    processor, model = load_model()

                    inputs = processor(
                        images=image,
                        return_tensors="pt"
                    )

                    with torch.no_grad():
                        outputs = model(**inputs)

                        probabilities = torch.softmax(
                            outputs.logits,
                            dim=-1
                        )[0]

                    top_scores, top_indices = torch.topk(
                        probabilities,
                        min(3, probabilities.numel())
                    )

                    category = model.config.id2label[
                        top_indices[0].item()
                    ]

                    category = category.replace("-", " ").title()

                    confidence = top_scores[0].item() * 100

                    tip = disposal_tips.get(
                        category,
                        "Check your local waste disposal guidelines."
                    )

                    result = {
                        "filename": uploaded_file.name,
                        "category": category,
                        "confidence": round(confidence, 2),
                        "tip": tip,
                        "top_predictions": [
                            (
                                model.config.id2label[index.item()]
                                .replace("-", " ")
                                .title(),
                                score.item() * 100
                            )
                            for score, index in zip(
                                top_scores,
                                top_indices
                            )
                        ]
                    }

                    st.session_state.last_result = result

                    st.session_state.history.insert(
                        0,
                        {
                            "timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "filename": uploaded_file.name,
                            "category": category,
                            "confidence": round(confidence, 2)
                        }
                    )

            except Exception as e:
                st.error(f"Classification error: {e}")

    result = st.session_state.last_result

    if result is not None:

        st.markdown("### 📊 Classification Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">PREDICTED CATEGORY</div>
                <div class="result-value">{result["category"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">MODEL CONFIDENCE</div>
                <div class="result-value">{result["confidence"]:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### ♻️ Disposal Recommendation")

        st.markdown(
            f'<div class="tip-card">{result["tip"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown("### 🔎 Top 3 AI Predictions")

        for index, (label, confidence) in enumerate(
            result["top_predictions"],
            start=1
        ):
            st.write(f"**{index}. {label} — {confidence:.2f}%**")
            st.progress(float(confidence / 100))

        st.caption(
            "AI predictions are estimates and may not always be correct. "
            "Always check local disposal guidelines."
        )

    else:
        st.info("Upload an image and click Analyze to get started.")


# =====================================================
# WASTE CATEGORIES PAGE
# =====================================================

elif st.session_state.current_page == "Categories":

    st.markdown(
        '<div class="section-title">♻️ 12 Waste Categories</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The AI model supports the following waste categories."
    )

    col1, col2 = st.columns(2)

    for index, category in enumerate(disposal_tips.keys()):

        target_col = col1 if index % 2 == 0 else col2

        with target_col:

            emoji = category_emojis[category]
            description = category_descriptions[category]

            st.markdown(f"""
            <div class="category-card">
                <h4>{emoji} {category}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

    st.info(
        "These are the 12 labels supported by the current AI model."
    )


# =====================================================
# ADMIN DASHBOARD
# =====================================================

elif st.session_state.current_page == "Dashboard":

    st.markdown(
        '<div class="section-title">📈 Admin Dashboard</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.history

    if history:

        df = pd.DataFrame(history)

        total_images = len(df)
        category_counts = df["category"].value_counts()
        most_common = category_counts.idxmax()
        average_confidence = df["confidence"].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric("📸 Total Images", total_images)
        col2.metric("🏆 Most Common Category", most_common)
        col3.metric("🎯 Average Confidence", f"{average_confidence:.2f}%")

        st.divider()

        st.markdown("### 📊 Category-wise Statistics")

        chart_data = category_counts.rename_axis(
            "Waste Category"
        ).reset_index(name="Number of Images")

        st.bar_chart(
            chart_data,
            x="Waste Category",
            y="Number of Images"
        )

        st.markdown("### 📋 Classification Summary")

        summary = (
            df.groupby("category")
            .agg(
                Total_Images=("category", "count"),
                Average_Confidence=("confidence", "mean")
            )
            .reset_index()
        )

        summary["Average_Confidence"] = (
            summary["Average_Confidence"].round(2)
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No statistics available yet. Classify some images first."
        )


# =====================================================
# CLASSIFICATION HISTORY PAGE
# =====================================================

elif st.session_state.current_page == "History":

    st.markdown(
        '<div class="section-title">📚 Classification History</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.history

    if history:

        df = pd.DataFrame(history)

        st.write("All images classified during this session:")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Classification History",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="greenvision_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Classification History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    else:
        st.info("No classification history available yet.")


# =====================================================
# CSV REPORT PAGE
# =====================================================

elif st.session_state.current_page == "CSV Report":

    st.markdown(
        '<div class="section-title">📥 CSV Reports</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.history

    if history:

        df = pd.DataFrame(history)

        st.success("Your CSV reports are ready!")

        st.markdown("### 📄 Complete Classification Report")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Complete History",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="greenvision_complete_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        st.markdown("### 📊 Category Summary")

        summary = (
            df.groupby("category")
            .agg(
                Total_Images=("category", "count"),
                Average_Confidence=("confidence", "mean")
            )
            .reset_index()
        )

        summary["Average_Confidence"] = (
            summary["Average_Confidence"].round(2)
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Category Summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="greenvision_category_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.warning(
            "No classification records found. "
            "Classify an image first to generate reports."
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    🌿 GREENVISION<br>
    <span style="color:#078c65;">
        Reduce • Reuse • Recycle • Protect Our Planet
    </span>
</div>
""", unsafe_allow_html=True)
