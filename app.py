import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Air Temperature Prediction",
    page_icon="🌡️",
    layout="wide"
)


# CUSTOM ATTENTION LAYER

class ResidualTemporalAttentionV8(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):

        e = tf.tanh(
            tf.matmul(x, self.W) + self.b
        )

        a = tf.nn.softmax(
            e,
            axis=1
        )

        context = tf.reduce_sum(
            x * a,
            axis=1
        )

        last_state = x[:, -1, :]

        return context + last_state


# CUSTOM LOSS

def composite_loss_v8(y_true, y_pred):

    mae = tf.reduce_mean(
        tf.abs(y_true - y_pred)
    )

    huber = tf.keras.losses.Huber(
        delta=0.2
    )(y_true, y_pred)

    return 0.7 * mae + 0.3 * huber


# LOAD MODEL

@st.cache_resource
def load_prediction_model():

    model = load_model(
        "models/cnn_bilstm_residual_attention_temperature_v8.keras",
        custom_objects={
            "ResidualTemporalAttentionV8":
                ResidualTemporalAttentionV8,
            "composite_loss_v8":
                composite_loss_v8
        },
        compile=False
    )

    return model


# LOAD SCALERS

@st.cache_resource
def load_scalers():

    scaler_X = joblib.load(
        "scalers/scaler_X.pkl"
    )

    scaler_y = joblib.load(
        "scalers/scaler_y.pkl"
    )

    return scaler_X, scaler_y


# MODEL INPUT FEATURES

FEATURES = [
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "dew_point_temp_c",
    "solar_radiation_kj_m2",
    "humidity_percent",
    "pressure_station_mb",
    "wind_speed_mps",
    "height_m",
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month"
]


# COLUMN DESCRIPTIONS

COLUMN_DESCRIPTIONS = {

    "temp_lag_1": {
        "description":
            "Air temperature from the previous hour.",
        "example":
            "24.2",
        "format":
            "Decimal number (°C)",
        "range":
            "No universal hard limit; should represent a realistic air temperature."
    },

    "temp_lag_2": {
        "description":
            "Air temperature from two hours earlier.",
        "example":
            "23.4",
        "format":
            "Decimal number (°C)",
        "range":
            "No universal hard limit; should represent a realistic air temperature."
    },

    "temp_lag_3": {
        "description":
            "Air temperature from three hours earlier.",
        "example":
            "22.7",
        "format":
            "Decimal number (°C)",
        "range":
            "No universal hard limit; should represent a realistic air temperature."
    },

    "dew_point_temp_c": {
        "description":
            "Dew-point temperature of the air.",
        "example":
            "17.4",
        "format":
            "Decimal number (°C)",
        "range":
            "Should normally be less than or equal to air temperature."
    },

    "solar_radiation_kj_m2": {
        "description":
            "Solar radiation received during the observation period.",
        "example":
            "2860",
        "format":
            "Numeric value (kJ/m²)",
        "range":
            "Normally 0 or greater."
    },

    "humidity_percent": {
        "description":
            "Relative humidity of the air.",
        "example":
            "61",
        "format":
            "Numeric percentage",
        "range":
            "0 to 100%"
    },

    "pressure_station_mb": {
        "description":
            "Atmospheric pressure measured at the weather station.",
        "example":
            "888.4",
        "format":
            "Decimal number (mb)",
        "range":
            "Must be greater than 0."
    },

    "wind_speed_mps": {
        "description":
            "Wind speed measured at the weather station.",
        "example":
            "1.8",
        "format":
            "Decimal number (m/s)",
        "range":
            "Normally 0 or greater."
    },

    "height_m": {
        "description":
            "Elevation of the weather station above sea level.",
        "example":
            "1159.54",
        "format":
            "Decimal number (metres)",
        "range":
            "Numeric elevation value."
    },

    "latitude": {
        "description":
            "Geographic latitude of the weather station.",
        "example":
            "-15.7894",
        "format":
            "Decimal degrees",
        "range":
            "-90 to 90"
    },

    "longitude": {
        "description":
            "Geographic longitude of the weather station.",
        "example":
            "-47.9000",
        "format":
            "Decimal degrees",
        "range":
            "-180 to 180"
    },

    "hour_sin": {
        "description":
            "Sine-based cyclic representation of the hour of the day.",
        "example":
            "0.5000",
        "format":
            "Decimal number",
        "range":
            "-1 to 1"
    },

    "hour_cos": {
        "description":
            "Cosine-based cyclic representation of the hour of the day.",
        "example":
            "-0.8660",
        "format":
            "Decimal number",
        "range":
            "-1 to 1"
    },

    "doy_sin": {
        "description":
            "Sine-based cyclic representation of the day of the year.",
        "example":
            "0.7500",
        "format":
            "Decimal number",
        "range":
            "-1 to 1"
    },

    "doy_cos": {
        "description":
            "Cosine-based cyclic representation of the day of the year.",
        "example":
            "-0.6600",
        "format":
            "Decimal number",
        "range":
            "-1 to 1"
    },

    "month": {
        "description":
            "Calendar month of the observation.",
        "example":
            "6",
        "format":
            "Integer",
        "range":
            "1 to 12"
    }
}


# PAGE HEADER

st.title(
    "🌡️ Short-Term Air Temperature Prediction"
)

st.markdown(
    """
    ### Deep Learning-Based Weather Forecasting

    Predict short-term air temperature using a
    **Multi-Scale CNN + BiLSTM + Residual Temporal Attention**
    deep learning model.
    """
)


# PREDICTION INPUT INFORMATION

st.subheader(
    "🔍 Prediction Input Information"
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.markdown(
        "**Required Input Columns**"
    )

    st.write(
        "The uploaded CSV must contain all 16 model input columns."
    )

with info_col2:

    st.markdown(
        "**Column Order**"
    )

    st.write(
        "Column order does not matter. The application automatically selects the columns in the order required by the model."
    )

with info_col3:

    st.markdown(
        "**Number of Rows**"
    )

    st.write(
        "At least 24 rows are required. If more rows are provided, the latest 24 observations are used."
    )


# LOAD SCALER INFORMATION

scaler_X, scaler_y = load_scalers()


# CHECK SCALER FEATURE COUNT

if hasattr(scaler_X, "n_features_in_"):

    if scaler_X.n_features_in_ != len(FEATURES):

        st.error(
            f"The saved input scaler expects "
            f"{scaler_X.n_features_in_} features, but the application "
            f"is configured for {len(FEATURES)} features."
        )

        st.stop()


# GET TRAINING RANGES

training_ranges = {}

if (
    hasattr(scaler_X, "data_min_")
    and hasattr(scaler_X, "data_max_")
    and len(scaler_X.data_min_) == len(FEATURES)
):

    for index, feature in enumerate(FEATURES):

        training_ranges[feature] = (
            float(scaler_X.data_min_[index]),
            float(scaler_X.data_max_[index])
        )


# REQUIRED COLUMNS INFORMATION

st.markdown(
    "#### Required Model Input Columns"
)

st.caption(
    "Click a column name to view its description, expected format, "
    "valid range, and training-data range."
)


column_rows = [
    FEATURES[i:i + 4]
    for i in range(0, len(FEATURES), 4)
]


for row in column_rows:

    cols = st.columns(4)

    for col, column in zip(cols, row):

        with col:

            with st.popover(
                column,
                use_container_width=True
            ):

                information = COLUMN_DESCRIPTIONS[
                    column
                ]

                st.markdown(
                    f"### {column}"
                )

                st.write(
                    information["description"]
                )

                st.markdown(
                    f"**Example:** `{information['example']}`"
                )

                st.markdown(
                    f"**Expected format:** {information['format']}"
                )

                st.markdown(
                    f"**General valid range:** {information['range']}"
                )

                if column in training_ranges:

                    train_min, train_max = (
                        training_ranges[column]
                    )

                    st.markdown(
                        "**Training-data range:** "
                        f"`{train_min:.4f}` to `{train_max:.4f}`"
                    )

                    st.caption(
                        "Values outside this range can still be processed, "
                        "but they are outside the range observed by the "
                        "input scaler during training."
                    )


# TARGET COLUMN INFORMATION

st.markdown(
    """
    <div style="
        margin-top: 15px;
        margin-bottom: 10px;
        color: #F1F3F5;
        font-size: 14px;
        white-space: nowrap;
    ">
        <strong>Target column:</strong>
        <code style="
            background-color: #252A31;
            color: #E6E9ED;
            padding: 4px 8px;
            border-radius: 5px;
            white-space: nowrap;
        ">air_temp_c</code>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Air temperature target in degrees Celsius. "
    "It is used for displaying the recent temperature trend "
    "and is not passed as an input feature to the prediction model."
)


# FILE UPLOAD

st.subheader(
    "📁 Upload Processed Weather Data"
)

st.write(
    "Upload a processed weather CSV containing the required columns."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# PROCESS UPLOADED FILE

if uploaded_file is not None:

    try:

        # READ CSV

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"CSV uploaded successfully — {len(df):,} rows detected."
        )


        # REQUIRED COLUMNS

        required_columns = FEATURES + [
            "air_temp_c"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]


        # CHECK MISSING COLUMNS

        if missing_columns:

            st.error(
                "The uploaded CSV is missing the following required columns:"
            )

            st.write(
                missing_columns
            )

            st.stop()


        # CHECK NUMBER OF ROWS

        if len(df) < 24:

            st.warning(
                "At least 24 rows are required because the model uses a 24-observation input window."
            )

            st.stop()


        # DISPLAY UPLOADED DATA

        with st.expander(
            "🔍 View Uploaded Data"
        ):

            st.write(
                f"Dataset: {len(df):,} rows × {len(df.columns)} columns"
            )

            st.dataframe(
                df,
                use_container_width=True,
                height=500
            )


        # CHECK TARGET DATA TYPE

        target_numeric = pd.to_numeric(
            df["air_temp_c"],
            errors="coerce"
        )

        invalid_target_rows = (
            target_numeric.isna()
        )

        if invalid_target_rows.any():

            st.error(
                "The target column `air_temp_c` contains "
                "non-numeric or invalid values."
            )

            st.write(
                f"Invalid rows detected: "
                f"{invalid_target_rows.sum():,}"
            )

            st.stop()


        # CHECK INPUT DATA TYPES

        invalid_feature_columns = []

        for column in FEATURES:

            converted_column = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            original_missing = df[column].isna()

            invalid_values = (
                converted_column.isna()
                & ~original_missing
            )

            if invalid_values.any():

                invalid_feature_columns.append(
                    column
                )


        if invalid_feature_columns:

            st.error(
                "The following model input columns contain "
                "non-numeric values:"
            )

            st.write(
                invalid_feature_columns
            )

            st.write(
                "Please provide numeric values such as integers "
                "or decimal numbers."
            )

            st.stop()


        # CONVERT INPUT FEATURES TO NUMERIC

        for column in FEATURES:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        # CHECK MISSING VALUES

        missing_feature_counts = (
            df[FEATURES]
            .isnull()
            .sum()
        )

        missing_features = (
            missing_feature_counts[
                missing_feature_counts > 0
            ]
        )


        if len(missing_features) > 0:

            st.error(
                "The uploaded data contains missing values "
                "in the model input features."
            )

            st.write(
                missing_features
            )

            st.stop()


        # CHECK PHYSICAL RANGES

        range_errors = []


        if (
            (df["humidity_percent"] < 0)
            | (df["humidity_percent"] > 100)
        ).any():

            range_errors.append(
                "humidity_percent must be between 0 and 100."
            )


        if (
            (df["latitude"] < -90)
            | (df["latitude"] > 90)
        ).any():

            range_errors.append(
                "latitude must be between -90 and 90."
            )


        if (
            (df["longitude"] < -180)
            | (df["longitude"] > 180)
        ).any():

            range_errors.append(
                "longitude must be between -180 and 180."
            )


        if (
            (df["hour_sin"] < -1)
            | (df["hour_sin"] > 1)
        ).any():

            range_errors.append(
                "hour_sin must be between -1 and 1."
            )


        if (
            (df["hour_cos"] < -1)
            | (df["hour_cos"] > 1)
        ).any():

            range_errors.append(
                "hour_cos must be between -1 and 1."
            )


        if (
            (df["doy_sin"] < -1)
            | (df["doy_sin"] > 1)
        ).any():

            range_errors.append(
                "doy_sin must be between -1 and 1."
            )


        if (
            (df["doy_cos"] < -1)
            | (df["doy_cos"] > 1)
        ).any():

            range_errors.append(
                "doy_cos must be between -1 and 1."
            )


        if (
            (df["month"] < 1)
            | (df["month"] > 12)
        ).any():

            range_errors.append(
                "month must be an integer from 1 to 12."
            )


        if (
            df["solar_radiation_kj_m2"] < 0
        ).any():

            range_errors.append(
                "solar_radiation_kj_m2 should not be negative."
            )


        if (
            df["wind_speed_mps"] < 0
        ).any():

            range_errors.append(
                "wind_speed_mps should not be negative."
            )


        if (
            df["pressure_station_mb"] <= 0
        ).any():

            range_errors.append(
                "pressure_station_mb must be greater than 0."
            )


        # DISPLAY RANGE ERRORS

        if range_errors:

            st.error(
                "Some input values are outside the allowed "
                "physical or mathematical ranges."
            )

            for error in range_errors:

                st.write(
                    f"• {error}"
                )

            st.stop()


        # CHECK TRAINING DATA RANGE

        training_warnings = []


        for column in FEATURES:

            if column not in training_ranges:

                continue

            train_min, train_max = (
                training_ranges[column]
            )

            current_min = df[column].min()
            current_max = df[column].max()

            if (
                current_min < train_min
                or current_max > train_max
            ):

                training_warnings.append(
                    f"`{column}` contains values outside "
                    f"the training range "
                    f"({train_min:.4f} to {train_max:.4f})."
                )


        # DISPLAY TRAINING RANGE WARNINGS

        if training_warnings:

            with st.expander(
                "⚠️ Values outside training-data range"
            ):

                st.write(
                    "These values are mathematically valid, "
                    "but they are outside the range observed "
                    "when the input scaler was fitted."
                )

                st.write(
                    "The model can still process them, but "
                    "predictions may be less representative "
                    "of the training distribution."
                )

                for warning in training_warnings:

                    st.write(
                        f"• {warning}"
                    )


        # SELECT LATEST 24 OBSERVATIONS

        latest_24 = df.tail(24).copy()


        # SELECT FEATURES IN MODEL ORDER

        X_latest = latest_24[
            FEATURES
        ]


        # SCALE INPUT FEATURES

        X_scaled = scaler_X.transform(
            X_latest
        )


        # CREATE MODEL INPUT

        X_sequence = np.expand_dims(
            X_scaled,
            axis=0
        )


        # PREDICTION BUTTON

        if st.button(
            "🔮 Predict Temperature",
            type="primary"
        ):

            with st.spinner(
                "Loading deep learning model..."
            ):

                model = load_prediction_model()


            with st.spinner(
                "Running prediction..."
            ):

                # GENERATE PREDICTION

                prediction_scaled = model.predict(
                    X_sequence,
                    verbose=0
                )


                # CONVERT PREDICTION TO CELSIUS

                prediction_actual = (
                    scaler_y.inverse_transform(
                        prediction_scaled
                    )
                )

                predicted_temperature = float(
                    prediction_actual[0][0]
                )


            # PREDICTION RESULT

            st.success(
                "Prediction completed successfully!"
            )

            st.subheader(
                "🌡️ Temperature Prediction"
            )

            st.metric(
                "Predicted Next-Hour Temperature",
                f"{predicted_temperature:.2f} °C"
            )


            # RECENT TEMPERATURE TREND

            st.subheader(
                "📈 Recent Temperature Trend"
            )

            chart_data = latest_24[
                ["air_temp_c"]
            ].reset_index(drop=True)

            chart_data.index = np.arange(
                1,
                len(chart_data) + 1
            )

            chart_data.columns = [
                "Air Temperature (°C)"
            ]

            st.line_chart(
                chart_data
            )


            # PREDICTION DETAILS

            st.subheader(
                "📋 Prediction Details"
            )

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.write(
                    "**Observations used:** 24"
                )

                st.write(
                    "**Input features:** 16"
                )

            with detail_col2:

                st.write(
                    "**Prediction:** Next-hour air temperature"
                )

                st.write(
                    "**Target unit:** °C"
                )


            # FINAL INFORMATION

            st.markdown(
                """
                <div style="
                    background-color: #171A1F;
                    padding: 18px 20px;
                    border-radius: 10px;
                    border: 1px solid #30343B;
                    margin-top: 15px;
                    color: #AEB4BD;
                    font-size: 14px;
                    line-height: 1.7;
                ">

                The model selects the latest
                <strong style="color: #F1F3F5;">
                24 observations
                </strong>
                from the uploaded dataset. The 16 required feature
                columns are automatically selected in the exact order
                expected by the trained model, regardless of their
                order in the uploaded CSV.

                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as e:

        st.error(
            "An error occurred while processing the CSV."
        )

        st.exception(e)


# INITIAL MESSAGE

else:

    st.info(
        "👆 Upload your processed weather CSV to begin."
    )


# FOOTER

st.markdown("---")

st.caption(
    "Deep Learning-Based Short-Term Air Temperature Prediction "
    "| Multi-Scale CNN + BiLSTM + Residual Temporal Attention"
)