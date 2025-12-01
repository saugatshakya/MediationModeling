"""
Interactive Streamlit Frontend for Exam Score Prediction

Features:
- Drink selector with caffeine calculation
- Interactive timetable/activity planner (24-hour day)
- Feature inputs for all relevant variables
- Exam score prediction
- Optimization for target scores
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
from scipy.optimize import minimize
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pipeline import ExamScorePipeline

def normalize_inputs_to_training_distribution(features):
    """
    Normalize app inputs to match the training data distribution.
    This prevents the model from extrapolating to unseen data ranges.
    """
    # Training data statistics (from processed_data.csv)
    training_stats = {
        'study_hours_per_day': {'mean': 3.539, 'std': 1.463, 'min': 0.0, 'max': 8.3},
        'sleep_hours': {'mean': 6.468, 'std': 1.234, 'min': 3.2, 'max': 10.0},
        'exercise_frequency': {'mean': 3.041, 'std': 2.009, 'min': 0.0, 'max': 6.0},
        'attendance_percentage': {'mean': 84.185, 'std': 9.262, 'min': 56.0, 'max': 100.0},
        'caffeine_intake': {'mean': 3.596, 'std': 1.396, 'min': 1.0, 'max': 10.0},
        'sleep_quality': {'mean': 4.339, 'std': 1.447, 'min': 1.0, 'max': 9.4}
    }
    
    normalized = features.copy()
    
    for feature, stats in training_stats.items():
        if feature in normalized:
            # Normalize to training distribution using z-score transformation
            # This maps app inputs to the training data distribution
            value = normalized[feature]
            
            # First, clamp to reasonable app ranges
            if feature == 'study_hours_per_day':
                value = min(max(value, 0), 12)  # App allows 0-12 hours
            elif feature == 'sleep_hours':
                value = min(max(value, 0), 12)  # App allows 0-12 hours
            elif feature == 'exercise_frequency':
                value = min(max(value, 0), 10)  # App allows 0-10
            elif feature == 'attendance_percentage':
                value = min(max(value, 0), 100)  # 0-100%
            
            # Transform to training distribution
            # Use sigmoid-like transformation to map app range to training distribution
            if feature == 'study_hours_per_day':
                # Map 0-12 hours in app to training distribution (centered around 3.5)
                z_score = (value - 6) / 3  # Center app range around 6
                normalized[feature] = stats['mean'] + z_score * stats['std']
            elif feature == 'sleep_hours':
                # Map 0-12 hours in app to training distribution
                z_score = (value - 6) / 3
                normalized[feature] = stats['mean'] + z_score * stats['std']
            elif feature == 'exercise_frequency':
                # Map 0-10 in app to training distribution
                z_score = (value - 5) / 2.5
                normalized[feature] = stats['mean'] + z_score * stats['std']
            elif feature == 'attendance_percentage':
                # Map 0-100 in app to training distribution
                z_score = (value - 50) / 25
                normalized[feature] = stats['mean'] + z_score * stats['std']
            
            # Ensure final values are within training data bounds
            normalized[feature] = np.clip(normalized[feature], stats['min'], stats['max'])
    
    return normalized
    """Load the best trained model for predictions."""
    try:
        # Try to load the best performing model (Random Forest tuned)
        model_path = Path(__file__).parent.parent / 'artifacts' / 'model_randomforest_tuned_model.joblib'
        if model_path.exists():
            return joblib.load(model_path)
        
        # Fallback to baseline model
        model_path = Path(__file__).parent.parent / 'artifacts' / 'baseline_all_features_model.joblib'
        if model_path.exists():
            return joblib.load(model_path)
            
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return None


# Drink data with caffeine content
DRINKS_DATA = {
    "Espresso": {"serving_ml": 30, "caffeine_mg": 63},
    "Americano": {"serving_ml": 240, "caffeine_mg": 95},
    "Cappuccino": {"serving_ml": 240, "caffeine_mg": 75},
    "Latte": {"serving_ml": 240, "caffeine_mg": 75},
    "Drip/Filter Coffee": {"serving_ml": 240, "caffeine_mg": 130},
    "Instant Coffee": {"serving_ml": 240, "caffeine_mg": 100},
    "Thai Iced Coffee": {"serving_ml": 240, "caffeine_mg": 70},
    "Black Tea": {"serving_ml": 240, "caffeine_mg": 55},
    "Green Tea": {"serving_ml": 240, "caffeine_mg": 32},
    "Oolong Tea": {"serving_ml": 240, "caffeine_mg": 40},
    "White Tea": {"serving_ml": 240, "caffeine_mg": 22},
    "Matcha": {"serving_ml": 240, "caffeine_mg": 70},
    "Thai Iced Tea": {"serving_ml": 240, "caffeine_mg": 45},
    "Red Bull": {"serving_ml": 250, "caffeine_mg": 80},
    "Monster": {"serving_ml": 500, "caffeine_mg": 160},
    "M-150": {"serving_ml": 250, "caffeine_mg": 80},
    "Carabao": {"serving_ml": 330, "caffeine_mg": 106},
    "5-hour Energy": {"serving_ml": 57, "caffeine_mg": 200},
}


def load_model():
    """Load the best trained model with fallbacks."""
    try:
        # Try to load the best performing model (Random Forest tuned)
        model_path = Path(__file__).parent.parent / 'artifacts' / 'model_randomforest_tuned_model.joblib'
        if model_path.exists():
            pipeline = ExamScorePipeline.load(str(model_path))
            print(f"Loaded best model: {model_path}")
            return pipeline
        
        # Fallback to baseline model
        model_path = Path(__file__).parent.parent / 'artifacts' / 'baseline_all_features_model.joblib'
        if model_path.exists():
            pipeline = ExamScorePipeline.load(str(model_path))
            print(f"Loaded fallback model: {model_path}")
            return pipeline
            
    except Exception as e:
        print(f"Error loading model: {e}")
    
    print("No trained models found, using fallback")
    return None


def calculate_caffeine_intake(drinks_selection):
    """Calculate total caffeine from selected drinks."""
    total_caffeine = 0
    for drink, count in drinks_selection.items():
        if count > 0:
            total_caffeine += DRINKS_DATA[drink]["caffeine_mg"] * count
    return total_caffeine


def caffeine_to_scale(caffeine_mg):
    """Convert caffeine mg to 1-10 scale (50-500mg → 1-10)."""
    return np.clip((caffeine_mg - 50) / 50, 1, 10)


def create_timetable_ui():
    """Create interactive timetable for 24-hour activity planning."""
    st.subheader("📅 Daily Activity Timetable")
    st.write("Allocate your 24 hours across activities (must sum to 24):")
    
    activities = {
        "Sleep": {"icon": "😴", "default": 7.0},
        "Study": {"icon": "📚", "default": 5.0},
        "Social Media": {"icon": "📱", "default": 2.0},
        "Netflix/Entertainment": {"icon": "📺", "default": 2.0},
        "Exercise": {"icon": "🏃", "default": 1.0},
        "Part-time Job": {"icon": "💼", "default": 0.0},
        "Other": {"icon": "🎯", "default": 7.0},
    }
    
    cols = st.columns(2)
    activity_hours = {}
    
    for i, (activity, info) in enumerate(activities.items()):
        with cols[i % 2]:
            hours = st.slider(
                f"{info['icon']} {activity}",
                min_value=0.0,
                max_value=24.0,
                value=info['default'],
                step=0.5,
                key=f"timetable_{activity}"
            )
            activity_hours[activity] = hours
    
    total_hours = sum(activity_hours.values())
    
    # Show total and validation
    if total_hours != 24.0:
        st.warning(f"⚠️ Total hours: {total_hours:.1f}/24. Please adjust to equal 24 hours.")
    else:
        st.success(f"✅ Total hours: {total_hours:.1f}/24")
    
    # Visualize timetable
    fig = go.Figure(data=[go.Pie(
        labels=list(activity_hours.keys()),
        values=list(activity_hours.values()),
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    fig.update_layout(
        title="Daily Time Distribution",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return activity_hours


def create_feature_inputs(activity_hours, caffeine_scale=3.8):
    """Create input widgets for all features."""
    st.subheader("📊 Additional Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Derived from timetable
        sleep_hours = activity_hours.get("Sleep", 7.0)
        study_hours = activity_hours.get("Study", 5.0)
        social_hours = activity_hours.get("Social Media", 2.0)
        netflix_hours = activity_hours.get("Netflix/Entertainment", 2.0)
        
        st.write(f"**Sleep Hours:** {sleep_hours}")
        st.write(f"**Study Hours:** {study_hours}")
        st.write(f"**Social Media:** {social_hours}")
        st.write(f"**Netflix:** {netflix_hours}")
        
        # Calculate exercise frequency from hours (0-10 scale)
        exercise_hours = activity_hours.get("Exercise", 1.0)
        exercise_frequency = min(10, int(exercise_hours * 2))  # 5 hours = 10/10
        st.write(f"**Exercise Frequency:** {exercise_frequency}/10")
        
        # Calculate sleep quality
        temp_sleep_quality = (
            0.5 * sleep_hours +
            0.3 * exercise_frequency +
            0.2 * 7 +  # Using default mental health
            -0.3 * caffeine_scale +  # Use actual caffeine intake (negative impact)
            0.5 * 1 +  # Using default good diet
            0.3 * 1 +  # Using default good internet
            np.random.normal(0, 0.3)  # Reduced noise for consistency
        )
        temp_sleep_quality = np.clip(temp_sleep_quality, 1, 10).round(1)
        st.write(f"**Calculated Sleep Quality:** {temp_sleep_quality}/10")
        
        # Additional inputs
        attendance = st.slider("Attendance %", 0, 100, 90, key="attendance")
        
    with col2:
        # Use default values for mental health, diet, and internet quality
        mental_health = 7  # Default good mental health
        diet_quality = "Good"  # Default good diet
        internet_quality = "Good"  # Default good internet
        
        part_time_job = "Yes" if activity_hours.get("Part-time Job", 0) > 0 else "No"
        st.write(f"**Part-time Job:** {part_time_job}")
        
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0, key="gender")
        age = st.number_input("Age", min_value=15, max_value=30, value=20, key="age")
    
    # Map categorical to numeric for model
    diet_map = {'Poor': -1, 'Fair': 0, 'Good': 1}
    internet_map = {'Poor': -1, 'Average': 0, 'Good': 1}
    
    # Calculate sleep_quality using the formula
    sleep_quality = (
        0.5 * sleep_hours +
        0.3 * exercise_frequency +
        0.2 * mental_health -
        0.3 * caffeine_scale +  # Use actual caffeine intake
        0.5 * diet_map[diet_quality] +
        0.3 * internet_map[internet_quality] +
        np.random.normal(0, 0.3)  # Reduced noise for more consistent results
    )
    sleep_quality = np.clip(sleep_quality, 1, 10).round(1)
    
    features = {
        'sleep_hours': sleep_hours,
        'study_hours_per_day': study_hours,
        'social_media_hours': social_hours,
        'netflix_hours': netflix_hours,
        'exercise_frequency': exercise_frequency,
        'sleep_quality': sleep_quality,
        'attendance_percentage': attendance,
        'mental_health_rating': mental_health,
        'diet_quality': diet_quality,
        'diet_score': diet_map[diet_quality],
        'internet_quality': internet_quality,
        'internet_score': internet_map[internet_quality],
        'part_time_job': part_time_job,
        'part_job_score': 1 if part_time_job == "Yes" else 0,
        'gender': gender,
        'age': age
    }
    
    return features


def calculate_stress_and_focus(features, caffeine_intake_scale):
    """Calculate stress and focus proxies using the formulas from main.ipynb."""
    # Normalize features
    Sd = features['sleep_hours'] / 12.0  # Assuming max 12 hours
    Sq = features['sleep_quality'] / 10.0
    
    # Caffeine effects
    C_mg = 50 + 50 * caffeine_intake_scale
    fC = np.exp(-((C_mg - 200) ** 2) / (2 * (100 ** 2)))
    gC = max(0, (C_mg - 300) / 200)
    
    # Normalize inputs
    study = features['study_hours_per_day'] / 10.0
    social = features['social_media_hours'] / 10.0
    netflix = features['netflix_hours'] / 10.0
    exercise = features['exercise_frequency'] / 10.0
    mental = features['mental_health_rating'] / 10.0
    
    # Stress calculation
    stress = (
        gC - Sq +
        0.4 * social +
        0.3 * netflix +
        0.2 * study +
        0.3 * features['part_job_score'] -
        0.3 * exercise -
        0.4 * mental -
        0.2 * features['diet_score'] -
        0.2 * features['internet_score']
    )
    
    # Focus calculation
    focus = (
        Sd + Sq - stress + fC +
        0.4 * exercise +
        0.4 * mental +
        0.3 * (features['attendance_percentage'] / 100) +
        0.2 * features['diet_score'] +
        0.2 * features['internet_score'] -
        0.3 * social -
        0.2 * netflix -
        0.2 * features['part_job_score']
    )
    
    # Rescale to 1-10
    stress_scaled = np.clip((stress + 2) * 2.5, 1, 10)  # Rough rescaling
    focus_scaled = np.clip((focus + 1) * 1.8, 1, 10)
    
    return stress_scaled, focus_scaled


def predict_exam_score(features, caffeine_intake_scale, stress, focus, model=None):
    """Predict exam score using the trained model or fallback to formula."""
    
    if model is not None:
        try:
            # Prepare features DataFrame for model prediction
            # Only include features that were used during training (from main.ipynb)
            feature_dict = {
                'study_hours_per_day': features['study_hours_per_day'],
                'sleep_hours': features['sleep_hours'],
                'social_media_hours': features['social_media_hours'],
                'netflix_hours': features['netflix_hours'],
                'exercise_frequency': features['exercise_frequency'],
                'sleep_quality': features['sleep_quality'],
                'attendance_percentage': features['attendance_percentage'],
                'caffeine_intake': caffeine_intake_scale,
                'stress_proxy': stress,
                'focus_proxy': focus,
            }
            
            # Create DataFrame with single row
            X = pd.DataFrame([feature_dict])
            
            # Predict using trained model
            raw_score = model.predict(X)[0]
            
            # Use the model's prediction directly, just ensure it's reasonable
            exam_score = np.clip(raw_score, 0, 100)
            
            return exam_score
            
        except Exception as e:
            st.warning(f"⚠️ Model prediction failed: {str(e)}. Using fallback formula.")
    
    # Fallback: balanced prediction requiring cognitive harmony
    # Study effort (max 40 points, but requires good cognitive state)
    study_base = min(features['study_hours_per_day'] * 4, 40)
    
    # Cognitive balance factor (0-1): requires both high focus AND low stress
    cognitive_balance = min(focus / 10, (10 - stress) / 10)  # Harmonic mean approach
    
    # Apply cognitive balance to study effort (max multiplier 1.0)
    study_component = study_base * (0.7 + cognitive_balance * 0.3)  # 70-100% of study effort
    
    # Other factors with diminishing returns
    attendance_component = min(features['attendance_percentage'] * 0.2, 20)  # Max 20 points
    sleep_component = min((features['sleep_quality'] + features['sleep_hours']) * 0.8, 25)  # Max 25 points
    
    # Total with balance requirement - perfect cognitive state needed for high scores
    base_score = study_component + attendance_component + sleep_component
    
    # Final cap at 92 to ensure true excellence requires perfect balance
    exam_score = min(base_score, 92)
    
    return exam_score


def load_inverse_model():
    """Load or create a model that predicts study hours needed for target score."""
    try:
        # Try to load pre-trained inverse model
        model_path = Path(__file__).parent.parent / 'artifacts' / 'inverse_study_hours_model.joblib'
        if model_path.exists():
            return joblib.load(model_path)
    except:
        pass
    
    # If no pre-trained model, create a simple rule-based predictor
    # This will be replaced with a proper ML model in production
    return None


def predict_study_hours_for_target(target_score, fixed_features, caffeine_scale, model=None):
    """
    Use the forward model with optimization to find study hours needed for target score.
    This is more reliable than training an inverse model.
    """
    # Ensure target score doesn't exceed 100
    target_score = min(target_score, 100)
    
    if model is None:
        # Fallback to simple rule-based
        base_hours = 2.0 + (min(target_score, 100) - 50) * 0.08
        adjustments = (
            -max(0, (fixed_features['sleep_quality'] - 5) * 0.1) +  # Good sleep reduces needs
            -max(0, (fixed_features['attendance_percentage'] - 80) / 200) +  # Good attendance helps
            -fixed_features['exercise_frequency'] / 20 +  # Exercise helps slightly
            caffeine_scale / 20  # Caffeine increases needs
        )
        return max(1.0, min(base_hours + adjustments, 8.0))
    
    # Use forward model with binary search to find optimal study hours
    def score_for_study_hours(study_hours):
        """Calculate predicted score for given study hours."""
        test_features = fixed_features.copy()
        test_features['study_hours_per_day'] = study_hours
        
        # Normalize the features
        normalized = normalize_inputs_to_training_distribution(test_features)
        
        # Calculate stress and focus
        stress, focus = calculate_stress_and_focus(normalized, caffeine_scale)
        
        # Get prediction
        predicted_score = predict_exam_score(normalized, caffeine_scale, stress, focus, model)
        return predicted_score
    
    # Binary search for study hours that give target score
    low, high = 0.0, 8.3  # Training data range
    best_hours = 4.0  # Default
    best_diff = float('inf')
    
    # Try a range of study hours
    for hours in np.linspace(0.5, 8.0, 20):
        score = score_for_study_hours(hours)
        diff = abs(score - target_score)
        if diff < best_diff:
            best_diff = diff
            best_hours = hours
    
    # Fine-tune with smaller steps around the best
    for offset in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]:
        hours = max(0, min(best_hours + offset, 8.3))
        score = score_for_study_hours(hours)
        diff = abs(score - target_score)
        if diff < best_diff:
            best_diff = diff
            best_hours = hours
    
    return round(best_hours, 1)


def optimize_for_target_score(current_features, current_caffeine, target_score, model=None):
    """Predict study hours needed to achieve target exam score."""
    # Ensure target score doesn't exceed 100
    target_score = min(target_score, 100)
    
    st.subheader("🎯 Study Hours Needed for Target Score")
    
    # Get current predicted score
    normalized_current = normalize_inputs_to_training_distribution(current_features)
    current_score_estimate = predict_exam_score(
        normalized_current,
        caffeine_to_scale(current_caffeine),
        *calculate_stress_and_focus(normalized_current, caffeine_to_scale(current_caffeine)),
        model
    )
    
    st.write(f"**Current predicted score:** {current_score_estimate:.1f}/100")
    st.write(f"**Target score:** {min(target_score, 100)}/100")
    
    if abs(current_score_estimate - target_score) < 2:
        st.success("✅ You're already very close to your target!")
        return
    
    # Predict required study hours
    required_hours = predict_study_hours_for_target(
        target_score, current_features, caffeine_to_scale(current_caffeine), model
    )
    
    current_study_hours = current_features['study_hours_per_day']
    
    st.write(f"**Current study hours:** {current_study_hours:.1f} hours/day")
    st.write(f"**Recommended study hours:** {required_hours:.1f} hours/day")
    
    if required_hours > current_study_hours:
        increase = required_hours - current_study_hours
        st.info(f"📈 **Increase study time by {increase:.1f} hours/day** to reach {target_score}")
        
        # Additional recommendations
        st.write("**Additional tips to reach your target:**")
        if current_features['sleep_quality'] < 7:
            st.write("- 😴 Improve sleep quality (aim for 7+/10)")
        if current_features['attendance_percentage'] < 90:
            st.write("- 🎓 Increase attendance (aim for 90%+)")
        if current_caffeine > 400:  # If high caffeine
            st.write("- ☕ Reduce caffeine intake for better focus")
            
    else:
        decrease = current_study_hours - required_hours
        st.success(f"📉 **You can reduce study time by {decrease:.1f} hours/day** and still achieve {target_score}")
        
        st.write("**You have room to:**")
        st.write("- 🏃 Add more exercise time")
        st.write("- 😴 Get more sleep")
        st.write("- 🎯 Pursue other interests")


def main():
    st.set_page_config(
        page_title="Exam Score Predictor",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Exam Score Prediction System")
    st.write("Predict your exam score based on lifestyle, study habits, and caffeine intake!")
    
    # Load model once and cache it
    @st.cache_resource
    def get_model():
        return load_model()
    
    model = get_model()
    
    # Show model status
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.warning("⚠️ Using fallback prediction formula")
    
    # Sidebar for drink selection
    st.sidebar.header("☕ Caffeine Intake Calculator")
    st.sidebar.write("Select your daily drinks:")
    
    drinks_selection = {}
    total_caffeine = 0
    
    for drink, info in DRINKS_DATA.items():
        count = st.sidebar.number_input(
            f"{drink} ({info['caffeine_mg']}mg / {info['serving_ml']}ml)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key=f"drink_{drink}"
        )
        drinks_selection[drink] = count
        total_caffeine += info['caffeine_mg'] * count
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Caffeine", f"{total_caffeine} mg")
    caffeine_scale = caffeine_to_scale(total_caffeine)
    st.sidebar.metric("Caffeine Scale (1-10)", f"{caffeine_scale:.1f}")
    
    # Health recommendations based on caffeine
    if total_caffeine > 400:
        st.sidebar.warning("⚠️ High caffeine intake! Consider reducing.")
    elif total_caffeine > 200:
        st.sidebar.info("ℹ️ Moderate caffeine intake.")
    else:
        st.sidebar.success("✅ Low to moderate caffeine.")
    
    # Main content
    tab1, tab2 = st.tabs(["📅 Timetable & Prediction", "🎯 Optimize"])
    
    with tab1:
        # Timetable
        activity_hours = create_timetable_ui()
        
        # Only proceed if timetable is valid
        if sum(activity_hours.values()) == 24.0:
            # Feature inputs
            features = create_feature_inputs(activity_hours, caffeine_scale)
            
            # Add caffeine to features
            features['caffeine_intake'] = caffeine_scale
            
            # Calculate stress and focus
            stress, focus = calculate_stress_and_focus(features, caffeine_scale)
            features['stress_proxy'] = stress
            features['focus_proxy'] = focus
            
            # Display calculated metrics
            st.subheader("📈 Calculated Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Caffeine Intake", f"{caffeine_scale:.1f}/10", 
                         delta=f"{total_caffeine} mg")
            with col2:
                st.metric("Stress Level", f"{stress:.1f}/10",
                         delta="Lower is better" if stress > 5 else "Good")
            with col3:
                st.metric("Focus Level", f"{focus:.1f}/10",
                         delta="Higher is better" if focus < 7 else "Good")
            
            # Predict exam score
            if st.button("🎯 Predict Exam Score", type="primary"):
                with st.spinner("Calculating prediction..."):
                    # Normalize inputs to training data distribution to prevent model extrapolation
                    normalized_features = normalize_inputs_to_training_distribution(features)
                    
                    exam_score = predict_exam_score(normalized_features, caffeine_scale, stress, focus, model)
                    
                    st.success(f"### Predicted Exam Score: {exam_score:.1f}/100")
                    
                    # Score interpretation
                    if exam_score >= 80:
                        st.balloons()
                        st.success("🌟 Excellent! Keep up the great work!")
                    elif exam_score >= 70:
                        st.info("👍 Good performance! Room for improvement.")
                    elif exam_score >= 60:
                        st.warning("⚠️ Passing, but consider optimizing your schedule.")
                    else:
                        st.error("❌ Below average. Check optimization suggestions!")
                    
                    # Store in session state for optimization
                    st.session_state['current_score'] = exam_score
                    st.session_state['current_features'] = features
                    st.session_state['current_caffeine'] = total_caffeine
    
    with tab2:
        st.header("🎯 Score Optimization")
        
        if 'current_score' in st.session_state:
            st.write(f"**Current Predicted Score:** {st.session_state['current_score']:.1f}/100")
            
            target_score = st.slider(
                "Target Exam Score",
                min_value=0,
                max_value=100,
                value=min(int(st.session_state['current_score']) + 10, 100),
                step=1
            )
            
            if st.button("Generate Recommendations"):
                optimize_for_target_score(
                    st.session_state['current_features'],
                    st.session_state['current_caffeine'],
                    target_score,
                    model
                )
        else:
            st.info("👈 Please make a prediction first in the 'Timetable & Prediction' tab.")


if __name__ == '__main__':
    main()
