import streamlit as st
import pandas as pd
import numpy as np
import io
import statistics
import math

# --- Page Config (ตั้งค่าหน้าเว็บ) ---
st.set_page_config(
    page_title="Intania Major Predictor",
    page_icon="⚙️",
    layout="wide"
)

# --- CSS Styling (แต่งสวย) ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #9d2449; /* Chula Pink */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. ข้อมูลฝังในโค้ด (Embedded Data)
# ==========================================
criteria_csv = """
รายวิชา (Subject),หน่วยกิต,โยธา,ไฟฟ้า,เครื่องกล,ยานยนต์,อุตสาหการ,เคมี,ปิโตรเลียม,สิ่งแวดล้อม,สำรวจ,โลหการ,นิวเคลียร์,คอมพิวเตอร์,ธรณี
ENG DRAW FUND,3,3,3,3,3,4,3,3,3,3,3,3,3,3
ENG MATERIALS,3,3,3,3,3,4,3,3,3,3,3,3,3,3
COMP PROG,3,3,3,3,3,5,3,4,3,6,3,3,9,4
EXPLORING ENG WORLD,3,3,3,3,3,1,3,3,3,3,3,3,3,3
CALCULUS I,3,3,6,3,3,5,3,3,3,6,3,3,3,3
CALCULUS II,3,3,6,3,3,5,3,3,3,6,3,3,3,3
GEN PHYS I,3,3,3,3,3,2,3,3,3,3,3,3,3,3
GEN PHYS II,3,3,6,3,3,2,3,3,3,3,3,3,3,3
GEN CHEM,3,3,3,3,3,2,3,3,3,3,3,3,3,3
EXP ENG I,3,3,3,3,3,3,3,4,3,3,3,3,3,4
EXP ENG II,3,3,3,3,3,3,3,4,3,3,3,3,3,4
GEN CHEM LAB,1,1,1,1,1,1,1,1,2,1,1,1,1,1
GEN PHYS LAB I,1,1,1,1,1,1,1,1,2,1,1,1,1,1
GEN PHYS LAB II,1,1,1,1,1,1,1,1,2,1,1,1,1,1
"""

history_csv = """
สาขา (Dept),ปีการศึกษา (Year),คะแนนต่ำสุด (Min Score),จำนวนรับภาครวม (Capacity),จำนวนคนเลือก (Demand),รับล่วงหน้า (Direct)
CE,2567,66.0,90,90,10
CE,2565,76.0,90,90,10
CE,2564,74.5,91,91,9
EE,2567,120.0,119,119,1
EE,2565,124.0,120,120,0
EE,2564,110.0,119,119,1
ME,2567,90.0,85,85,0
ME,2566,79.5,90,90,0
ME,2565,93.5,84,84,1
AE,2567,77.5,15,15,0
AE,2565,76.0,15,15,0
IE,2567,100.0,80,80,5
IE,2565,116.5,80,80,5
CHE,2567,73.0,40,40,0
CHE,2566,57.5,42,42,0
CHE,2565,66.5,40,40,0
PE,2567,88.5,10,10,0
PE,2565,97.5,10,10,0
ENV,2567,77.0,9,9,31
ENV,2565,81.5,13,13,26
SV,2567,103.0,3,3,37
SV,2566,55.5,6,3,36
MT,2567,45.0,45,44,10
MT,2565,49.5,27,27,22
CP,2567,115.5,42,42,98
CP,2566,115.0,57,57,91
CP,2565,153.0,37,37,83
GE,2566,40.5,1,1,28
NU,2566,75.0,1,1,22
"""

# ==========================================
# 2. Logic การคำนวณ (เหมือนเดิมแต่ปรับให้รองรับ Streamlit)
# ==========================================
@st.cache_data
def load_data():
    criteria = pd.read_csv(io.StringIO(criteria_csv))
    history = pd.read_csv(io.StringIO(history_csv))
    
    dept_map = {
        'โยธา': 'CE', 'ไฟฟ้า': 'EE', 'เครื่องกล': 'ME', 'ยานยนต์': 'AE',
        'อุตสาหการ': 'IE', 'เคมี': 'CHE', 'ปิโตรเลียม': 'PE', 'ธรณี': 'GE',
        'สิ่งแวดล้อม': 'ENV', 'สำรวจ': 'SV', 'โลหการ': 'MT',
        'นิวเคลียร์': 'NU', 'คอมพิวเตอร์': 'CP'
    }
    criteria.rename(columns=dept_map, inplace=True)
    history_clean = history.dropna(subset=['คะแนนต่ำสุด (Min Score)'])
    return criteria, history_clean

criteria, history_clean = load_data()

subject_info = {
    'CALCULUS I': 3, 'CALCULUS II': 3, 'GEN PHYS I': 3, 'GEN PHYS II': 3, 'GEN CHEM': 3,
    'COMP PROG': 3, 'ENG DRAW FUND': 3, 'ENG MATERIALS': 3, 'EXPLORING ENG WORLD': 3,
    'EXP ENG I': 3, 'EXP ENG II': 3, 'GEN CHEM LAB': 1, 'GEN PHYS LAB I': 1, 'GEN PHYS LAB II': 1
}

def grade_to_point(g_str):
    mapping = {'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5, 'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0}
    return mapping.get(g_str, None)

def get_competition_factor(dept):
    dept_data = history_clean[history_clean['สาขา (Dept)'] == dept].copy()
    if dept_data.empty: return 1.0, 0
    try:
        dept_data['Available'] = dept_data['จำนวนรับภาครวม (Capacity)'] - dept_data['รับล่วงหน้า (Direct)']
        dept_data['Available'] = dept_data['Available'].apply(lambda x: x if x > 0 else 1)
        dept_data['Rate'] = dept_data['จำนวนคนเลือก (Demand)'] / dept_data['Available']
        return dept_data['Rate'].mean(), dept_data['รับล่วงหน้า (Direct)'].mean()
    except:
        return 1.0, 0

def calculate_statistical_chance(user_score, historical_scores, comp_rate):
    if len(historical_scores) < 2: return 0, 0, 0
    mu = statistics.mean(historical_scores)
    sigma = statistics.stdev(historical_scores)
    if sigma < 5: sigma = 5.0

    n = len(historical_scores)
    prediction_error = sigma * math.sqrt(1 + (1/n))
    z_score = (user_score - mu) / prediction_error
    probability = (1.0 + math.erf(z_score / math.sqrt(2.0))) / 2.0
    chance_percent = probability * 100

    penalty = 0
    if comp_rate > 2.0: penalty = 15
    elif comp_rate > 1.5: penalty = 10
    elif comp_rate > 1.2: penalty = 5
    elif comp_rate < 0.8 and comp_rate > 0: penalty = -5

    final_chance = max(0, min(99, int(chance_percent - penalty)))
    
    cv = (sigma / mu) * 100 if mu > 0 else 0
    error_margin = max(3, min(20, int(cv * 1.5)))
    
    return final_chance, sigma, error_margin

# ==========================================
# 3. ส่วน UI (หน้าตาเว็บ)
# ==========================================
st.title("⚙️ Intania Major Predictor 2025")
st.write("โปรแกรมประเมินโอกาสเลือกภาควิชา (Statistical Model + Competition Factor)")

# Sidebar: กรอกเกรด
with st.sidebar:
    st.header("📝 กรอกเกรดของคุณ")
    st.info("วิชาไหนเกรดยังไม่ออก ให้เลือก 'ยังไม่ออก'")
    
    user_inputs = {}
    grade_options = ["ยังไม่ออก", "A", "B+", "B", "C+", "C", "D+", "D", "F"]
    
    # จัดกลุ่มวิชาให้ดูง่าย
    st.subheader("วิชาหลัก")
    for sub in ['CALCULUS I', 'CALCULUS II', 'GEN PHYS I', 'GEN PHYS II', 'GEN CHEM', 'COMP PROG']:
        user_inputs[sub] = st.selectbox(sub, grade_options, index=0)
        
    st.subheader("วิชาพื้นฐาน & Lab")
    for sub in ['ENG DRAW FUND', 'ENG MATERIALS', 'EXPLORING ENG WORLD', 'EXP ENG I', 'EXP ENG II']:
        user_inputs[sub] = st.selectbox(sub, grade_options, index=0)
    for sub in ['GEN CHEM LAB', 'GEN PHYS LAB I', 'GEN PHYS LAB II']:
        user_inputs[sub] = st.selectbox(sub, grade_options, index=0)

# คำนวณ GPAX
current_grades = {k: grade_to_point(v) for k, v in user_inputs.items()}
total_p, total_c = 0, 0
for sub, grade in current_grades.items():
    if grade is not None:
        total_p += grade * subject_info.get(sub, 3)
        total_c += subject_info.get(sub, 3)
gpax = total_p / total_c if total_c > 0 else 0.0

# แสดงผล GPAX ด้านบน
col1, col2 = st.columns(2)
with col1:
    st.metric("GPAX (เฉพาะวิชาที่มีเกรด)", f"{gpax:.2f}")
with col2:
    st.metric("หน่วยกิตที่คำนวณ", f"{total_c}")

st.divider()

# ปุ่มคำนวณ
if st.button("🚀 คำนวณความน่าจะเป็น", type="primary"):
    
    results = []
    target_depts = list(criteria.columns[2:]) # เอาทุกภาค

    for dept in target_depts:
        # คำนวณคะแนนรวม
        total_score = 0
        for user_sub, grade in current_grades.items():
            calc_grade = grade if grade is not None else 0.0
            crit_name = user_sub # ชื่อตรงกันอยู่แล้วจาก map
            
            weight_row = criteria[criteria['รายวิชา (Subject)'] == crit_name]
            if not weight_row.empty:
                w = weight_row[dept].values[0]
                total_score += calc_grade * w

        if total_score > 1000: total_score /= 100 # Auto-scale fix

        # ดึงสถิติ
        stats = history_clean[history_clean['สาขา (Dept)'] == dept]
        score_list = stats['คะแนนต่ำสุด (Min Score)'].tolist()

        if len(score_list) >= 2:
            avg_rate, avg_direct = get_competition_factor(dept)
            chance, sigma, error_margin = calculate_statistical_chance(total_score, score_list, avg_rate)
            
            # Icon การแข่งขัน
            comp_icon = "🟢" if avg_rate < 0.8 else ("🔴" if avg_rate > 1.5 else "🟡")
            
            results.append({
                'ภาควิชา': dept,
                'คะแนนรวม': total_score,
                'โอกาสติด (%)': chance,
                'Error (±)': f"{error_margin}%",
                'ความผันผวน': f"±{sigma:.1f}",
                'การแข่งขัน': f"{avg_rate:.1f}:1 {comp_icon}",
                'Min Cut-off': min(score_list),
                'Max Cut-off': max(score_list)
            })
        else:
            results.append({
                'ภาควิชา': dept, 'คะแนนรวม': total_score, 'โอกาสติด (%)': 0, 
                'Error (±)': "-", 'ความผันผวน': "-", 'การแข่งขัน': "-", 'Min Cut-off': 0, 'Max Cut-off': 0
            })

    # แปลงเป็น DataFrame และแสดงผล
    df = pd.DataFrame(results)
    df = df.sort_values(by='โอกาสติด (%)', ascending=False)
    
    # ใช้ Column Config ของ Streamlit แต่งตารางสวยๆ
    st.subheader("📊 ผลลัพธ์การเปรียบเทียบ")
    
    st.dataframe(
        df,
        column_config={
            "โอกาสติด (%)": st.column_config.ProgressColumn(
                "โอกาสติด",
                help="คำนวณจาก Z-Score เทียบกับสถิติเก่า",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
            "คะแนนรวม": st.column_config.NumberColumn(format="%.2f"),
            "Min Cut-off": st.column_config.NumberColumn(format="%.0f"),
            "Max Cut-off": st.column_config.NumberColumn(format="%.0f"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.info("""
    **คำอธิบาย:**
    * **โอกาสติด:** แถบสีบอกความน่าจะเป็น (ยิ่งยาวยิ่งดี)
    * **การแข่งขัน:** อัตราส่วน คนสมัคร : ที่นั่ง (ถ้าแดง 🔥 แปลว่าคนแย่งเยอะ)
    * **Error:** ค่าบวกลบที่เป็นไปได้จากความผันผวนของคะแนน
    """)

else:
    st.write("👈 กรอกเกรดทางซ้ายมือ แล้วกดปุ่มคำนวณได้เลย")
