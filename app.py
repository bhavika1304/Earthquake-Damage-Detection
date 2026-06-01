import streamlit as st
import numpy as np
import tifffile as tiff
from PIL import Image
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    accuracy_score
)
from preprocessing.blur import apply_gaussian_blur
from preprocessing.brightness import apply_histogram_matching
from preprocessing.registration import register_images

from inference.predict import (
    load_model,
    predict_damage
)

from inference.overlay import (
    create_overlay
)

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Earthquake Damage Detection",
    layout="wide"
)

st.title("Earthquake Damage Detection System")

st.markdown(
    "Upload pre- and post-earthquake satellite images."
)

# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------

if "processed_pre" not in st.session_state:

    st.session_state.processed_pre = None

if "processed_post" not in st.session_state:

    st.session_state.processed_post = None

if "prediction_ready" not in st.session_state:

    st.session_state.prediction_ready = False

# ---------------------------------------------------
# Normalization
# ---------------------------------------------------

def normalize_rgb(img):

    img = img.astype(np.float32)

    for c in range(3):

        channel = img[:, :, c]

        p2 = np.percentile(channel, 2)
        p98 = np.percentile(channel, 98)

        channel = np.clip(channel, p2, p98)

        channel = (
            (channel - p2)
            / (p98 - p2 + 1e-8)
        )

        img[:, :, c] = channel

    img = (img * 255).astype(np.uint8)

    return img

# ---------------------------------------------------
# China Loader
# ---------------------------------------------------

def load_china_image(uploaded_file):

    img = np.array(
        Image.open(uploaded_file).convert("RGB")
    )

    return img

# ---------------------------------------------------
# Morocco Loader
# ---------------------------------------------------

def load_morocco_image(uploaded_file):
    import cv2
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    uploaded_file.seek(0)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read Morocco image.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ---------------------------------------------------
# Mexico Loader
# ---------------------------------------------------

def load_mexico_image(uploaded_file):

    import cv2

    data = np.frombuffer(
        uploaded_file.read(),
        np.uint8
    )

    uploaded_file.seek(0)

    img = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if img is None:

        st.error("Could not read Mexico image.")

        return None

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img

# ---------------------------------------------------
# Master Loader
# ---------------------------------------------------

def load_image(uploaded_file, region):

    try:

        if region == "China":

            return load_china_image(uploaded_file)

        elif region == "Morocco":

            return load_morocco_image(uploaded_file)

        elif region == "Mexico":

            return load_mexico_image(uploaded_file)

    except Exception as e:

        st.error(f"Error loading image: {e}")

        return None

# ---------------------------------------------------
# REGION SELECTION
# ---------------------------------------------------

region = st.selectbox(
    "Select Region",
    [
        "China",
        "Morocco",
        "Mexico"
    ]
)

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    pre_file = st.file_uploader(
        "Upload Pre-Earthquake Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

with col2:
    post_file = st.file_uploader(
        "Upload Post-Earthquake Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

with col3:
    gt_file = st.file_uploader(
        "Upload Ground Truth Mask",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

# ---------------------------------------------------
# Issue Selection
# ---------------------------------------------------

blur_target = None

issue = st.selectbox(
    "Select Quality Issue",
    [
        "--Choose from the options below--",
        "None",
        "Brightness Difference",
        "Blur",
        "Registration"
    ]
)

# ---------------------------------------------------
# Blur Selection
# ---------------------------------------------------

if issue == "Blur":

    blur_target = st.radio(
        "Which image contains blur?",
        [
            "Pre-Earthquake Image",
            "Post-Earthquake Image"
        ]
    )

if issue == "Brightness Difference":
    brightness_target = st.radio(
        "Which image is brighter?",
        [
            "Pre-Earthquake Image",
            "Post-Earthquake Image"
        ]
    )

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

if pre_file and post_file:

    # ---------------------------------------------------
    # Load Images
    # ---------------------------------------------------

    pre_img = load_image(
        pre_file,
        region
    )

    post_img = load_image(
        post_file,
        region
    )

    if pre_img is None or post_img is None:
        st.stop()

    # load ground truth if provided
    gt_img = None
    if gt_file is not None:
        try:
            import tifffile as tiff

            name = gt_file.name.lower()
            if name.endswith((".tif", ".tiff")):
                gt_raw = tiff.imread(gt_file)
            else:
                gt_raw = np.array(Image.open(gt_file).convert("L"))
            # resize to 256x256 and keep raw class values
            gt_img = np.array(
                Image.fromarray(gt_raw.astype(np.uint8))
                .resize((256, 256), Image.NEAREST)
            )
        except Exception as e:
            st.warning(f"Could not load ground truth: {e}")

    # ---------------------------------------------------
    # Display Original Images
    # ---------------------------------------------------

    st.subheader("Original Images")

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            pre_img,
            caption="Pre-Earthquake Image",
            width=500
        )

    with c2:

        st.image(
            post_img,
            caption="Post-Earthquake Image",
            width=500
        )

    # ---------------------------------------------------
    # Initialize
    # ---------------------------------------------------

    processed_pre = pre_img.copy()

    processed_post = post_img.copy()

    # ---------------------------------------------------
    # No preprocessing needed
    # ---------------------------------------------------

    if issue == "None":
        st.session_state.processed_pre = processed_pre

        st.session_state.processed_post = processed_post

        st.session_state.prediction_ready = True

    # ---------------------------------------------------
    # PREPROCESSING BUTTON
    # ---------------------------------------------------

    run_preprocess = False

    if issue != "None":

        run_preprocess = st.button(
            "Run Preprocessing"
        )

    if run_preprocess:

        with st.spinner("Applying preprocessing..."):

            # ---------------------------------------------------
            # Blur
            # ---------------------------------------------------

            if issue == "Blur":

                if blur_target == "Pre-Earthquake Image":

                    processed_post = apply_gaussian_blur(
                        post_img
                    )

                elif blur_target == "Post-Earthquake Image":

                    processed_pre = apply_gaussian_blur(
                        pre_img
                    )

            # ---------------------------------------------------
            # Brightness
            # ---------------------------------------------------

            elif issue == "Brightness Difference":

                if brightness_target == "Pre-Earthquake Image":

                    # pre is brighter — match pre to post's histogram

                    processed_pre = apply_histogram_matching(

                        pre_img,

                        post_img

                    )

                    processed_pre = np.clip(
                        processed_pre,
                        0,
                        255
                    ).astype(np.uint8)

                else:

                    # post is brighter — match post to pre's histogram

                    processed_post = apply_histogram_matching(

                        post_img,

                        pre_img

                    )

                    processed_post = normalize_rgb(processed_post)

            # ---------------------------------------------------
            # Registration
            # ---------------------------------------------------

            elif issue == "Registration":

                processed_post = register_images(
                    pre_img,
                    post_img
                )

                processed_post = np.clip(
                    processed_post,
                    0,
                    255
                ).astype(np.uint8)

        # ---------------------------------------------------
        # Save in Session State
        # ---------------------------------------------------

        st.session_state.processed_pre = processed_pre

        st.session_state.processed_post = processed_post

        # ---------------------------------------------------
        # Display Results
        # ---------------------------------------------------

        st.subheader("Preprocessing Results")

        c3, c4 = st.columns(2)

        with c3:

            st.image(
                processed_pre,
                caption="Processed Pre-Earthquake",
                width=500
            )

        with c4:

            st.image(
                processed_post,
                caption="Processed Post-Earthquake",
                width=500
            )

        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------

        quality_ok = st.radio(
            "Is the quality issue handled properly?",
            ["Yes", "No"]
        )

        if quality_ok == "Yes":

            st.success(
                "Preprocessing successful."
            )

            st.session_state.prediction_ready = True

        else:

            st.warning(
                "Please rerun preprocessing."
            )

    # ---------------------------------------------------
    # Prediction Section
    # ---------------------------------------------------

    if st.session_state.prediction_ready:

        if st.button("Run Damage Prediction"):

            with st.spinner("Running prediction..."):

                processed_pre = st.session_state.processed_pre

                processed_post = st.session_state.processed_post

                # ---------------------------------------------------
                # Load Model
                # ---------------------------------------------------

                model = load_model(region)

                # ---------------------------------------------------
                # Predict
                # ---------------------------------------------------

                pred_mask = predict_damage(
                    model,
                    processed_pre,
                    processed_post,
                    region
                )

                # ---------------------------------------------------
                # Overlay
                # ---------------------------------------------------

                h, w = pred_mask.shape[:2]

                post_disp = np.array(
                    Image.fromarray(processed_post).resize((w, h))
                )
                pre_disp = np.array(
                    Image.fromarray(processed_pre).resize((w, h))
                )

                if pre_disp.dtype != np.uint8:
                    pre_disp = (pre_disp * 255).clip(0, 255).astype(np.uint8)
                if post_disp.dtype != np.uint8:
                    post_disp = (post_disp * 255).clip(0, 255).astype(np.uint8)

                overlay = create_overlay(
                    post_disp,
                    pred_mask
                )

                # ---------------------------------------------------
                # Metrics
                # ---------------------------------------------------

                f1 = None
                iou = None
                precision = None
                recall = None
                accuracy = None

                if gt_img is not None:
                    # ---------------------------------------------------
                    # Match GT size to prediction
                    # ---------------------------------------------------

                    if gt_img.shape != pred_mask.shape:
                        gt_img = np.array(
                            Image.fromarray(
                                gt_img.astype(np.uint8)
                            ).resize(
                                (
                                    pred_mask.shape[1],
                                    pred_mask.shape[0]
                                ),
                                Image.NEAREST
                            )
                        )

                    gt_flat = gt_img.flatten()

                    pred_flat = pred_mask.flatten()

                    f1 = f1_score(
                        gt_flat,
                        pred_flat,
                        average="macro"
                    )

                    iou = jaccard_score(
                        gt_flat,
                        pred_flat,
                        average="macro"
                    )

                    precision = precision_score(
                        gt_flat,
                        pred_flat,
                        average="macro",
                        zero_division=0
                    )

                    recall = recall_score(
                        gt_flat,
                        pred_flat,
                        average="macro",
                        zero_division=0
                    )

                    accuracy = accuracy_score(
                        gt_flat,
                        pred_flat
                    )

            # ---------------------------------------------------
            # Display Results
            # ---------------------------------------------------

            st.subheader("Prediction Results")

            r1, r2, r3, r4, r5 = st.columns(5)

            with r1:
                st.image(
                    pre_disp,
                    caption="Pre-Earthquake",
                    width='stretch'
                )

            with r2:
                st.image(
                    post_disp,
                    caption="Post-Earthquake",
                    width='stretch'
                )

            with r3:
                if gt_img is not None:
                    gt_visual = np.zeros(
                        (gt_img.shape[0], gt_img.shape[1], 3),
                        dtype=np.uint8
                    )
                    gt_visual[gt_img == 1] = [255, 255, 0]
                    gt_visual[gt_img == 2] = [255, 0, 0]
                    st.image(
                        gt_visual,
                        caption="Ground Truth",
                        width='stretch'
                    )
                else:
                    st.info("No ground truth uploaded")

            with r4:
                visual_mask = np.zeros(
                    (pred_mask.shape[0], pred_mask.shape[1], 3),
                    dtype=np.uint8
                )
                visual_mask[pred_mask == 1] = [255, 255, 0]
                visual_mask[pred_mask == 2] = [255, 0, 0]
                st.image(
                    visual_mask,
                    caption="Predicted Mask",
                    width='stretch'
                )

            with r5:
                st.image(
                    overlay,
                    caption="Overlay on Post",
                    width='stretch'
                )

            # ---------------------------------------------------
            # Compact Legends
            # ---------------------------------------------------

            legend1, legend2, legend3 = st.columns(3)

            with legend1:

                st.markdown(
                    '<div style="display:flex;align-items:center;gap:10px;margin-top:10px;">'
                    '<div style="width:22px;height:22px;background-color:black;'
                    'border:1px solid white;border-radius:4px;"></div>'
                    '<span style="font-size:18px;color:white;">'
                    'Background / No Damage'
                    '</span></div>',
                    unsafe_allow_html=True
                )

            with legend2:

                st.markdown(
                    '<div style="display:flex;align-items:center;gap:10px;margin-top:10px;">'
                    '<div style="width:22px;height:22px;background-color:yellow;'
                    'border-radius:4px;"></div>'
                    '<span style="font-size:18px;color:white;">'
                    'Severe Damage'
                    '</span></div>',
                    unsafe_allow_html=True
                )

            with legend3:

                st.markdown(
                    '<div style="display:flex;align-items:center;gap:10px;margin-top:10px;">'
                    '<div style="width:22px;height:22px;background-color:red;'
                    'border-radius:4px;"></div>'
                    '<span style="font-size:18px;color:white;">'
                    'Complete Collapse'
                    '</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------
            # Metrics Display
            # ---------------------------------------------------

            st.markdown(
                """
                <style>
                div[data-testid="stMetric"] {
                    background-color: #111827;
                    border: 1px solid #333;
                    padding: 15px;
                    border-radius: 12px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            if f1 is not None and iou is not None:
                m1, m2, m3, m4, m5 = st.columns(5)

                with m1:
                    st.metric(
                        label="Accuracy",
                        value=f"{accuracy:.4f}"
                    )

                with m2:
                    st.metric(
                        label="Precision",
                        value=f"{precision:.4f}"
                    )

                with m3:
                    st.metric(
                        label="Recall",
                        value=f"{recall:.4f}"
                    )

                with m4:
                    st.metric(
                        label="F1 Score",
                        value=f"{f1:.4f}"
                    )

                with m5:
                    st.metric(
                        label="IoU",
                        value=f"{iou:.4f}"
                    )