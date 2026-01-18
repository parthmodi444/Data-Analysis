import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Advanced Customer Insights Dashboard",
    layout="wide"
)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
df["Recommend_Binary"] = df["Recommend"].map({"Yes": 1, "No": 0})

df["Age_Group"] = pd.cut(
    df["age"],
    bins=[17, 25, 35, 50, 100],
    labels=["18–25", "26–35", "36–50", "50+"]
)

# -----------------------------
# Sidebar Controls (Universal)
# -----------------------------
st.sidebar.header("🎛 Controls")

# Tab selection
tab_selection = st.sidebar.radio(
    "Select Dashboard View",
    ["📊 Analytics Dashboard", "🤖 ML Predictions"],
    key="main_tab_selector"
)

# Segment for multi-line trends, heatmaps, etc.
segment_option = st.sidebar.selectbox(
    "Segment By (applies to charts)",
    ["gender", "Payment", "Work", "neighbourhood"]
)

# Age filter (applies to all charts)
age_range = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (18, 45)
)

# Optional: filter by Satisfaction
satisfaction_filter = st.sidebar.multiselect(
    "Satisfied Status (optional)",
    options=df["Satisfied"].unique(),
    default=df["Satisfied"].unique()
)

# Optional: filter by Recommendation
recommend_filter = st.sidebar.multiselect(
    "Recommendation Status (optional)",
    options=df["Recommend"].unique(),
    default=df["Recommend"].unique()
)

# Optional: filter by Neighbourhood
neighbourhood_filter = st.sidebar.multiselect(
    "Neighbourhood (optional)",
    options=df["neighbourhood"].unique(),
    default=df["neighbourhood"].unique()
)

# Optional: filter by Last Purchase
last_purchase_filter = st.sidebar.multiselect(
    "Last Purchase Category (optional)",
    options=df["Last_Purchase"].unique(),
    default=df["Last_Purchase"].unique()
)

# -----------------------------
# Apply universal filters with error handling
# -----------------------------
filtered_df = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["Satisfied"].isin(satisfaction_filter)) &
    (df["Recommend"].isin(recommend_filter)) &
    (df["neighbourhood"].isin(neighbourhood_filter)) &
    (df["Last_Purchase"].isin(last_purchase_filter))
]

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Adjust filters to see charts.")
    st.stop()

# =====================================================
# MAIN CONTENT - CONDITIONAL BASED ON TAB SELECTION
# =====================================================

if tab_selection == "📊 Analytics Dashboard":
    # Original analytics dashboard code
    st.title("📊 Advanced Customer Behaviour Insights")

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Spend", f"{filtered_df['Total_Spend'].mean():.1f}")
    k2.metric("Avg Profit %", f"{filtered_df['Profit_Percentage'].mean():.1f}")
    k3.metric("Recommendation Rate",
            f"{filtered_df['Recommend_Binary'].mean() * 100:.1f}%")

    # =====================================================
    # GRAPH 1: BOXPLOT – Spend Distribution
    # =====================================================
    st.subheader("💰 Spending Distribution by Age Group")

    fig1, ax1 = plt.subplots()
    filtered_df.boxplot(
        column="Total_Spend",
        by="Age_Group",
        ax=ax1
    )
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Total Spend")
    plt.suptitle("")
    st.pyplot(fig1)

    st.markdown(
        "Insight: Box plots reveal not just averages but spending variability and outliers.\n\n"
        "Action: High-spread segments need different pricing strategies than stable ones."
    )



    # =====================================================
    # GRAPH 3: SATISFACTION → RECOMMENDATION
    # =====================================================
    st.subheader("⭐ Satisfaction Drives Recommendations")

    sat_rec = filtered_df.groupby("Satisfied")["Recommend_Binary"].mean()

    fig3, ax3 = plt.subplots()
    sat_rec.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Recommendation Rate")
    st.pyplot(fig3)

    st.markdown(
        "Insight: Satisfaction is the strongest predictor of advocacy.\n\n"
        "Action: Prioritize fixing pain points for partially satisfied users."
    )

    # =====================================================
    # GRAPH 4: MULTI-LINE QUARTERLY TREND
    # =====================================================
    st.subheader(f"📈 Quarterly Spending Trend by {segment_option}")

    fig4, ax4 = plt.subplots()

    for segment, group in filtered_df.groupby(segment_option):
        quarter_means = group[
            ["First_Qt", "Second_Qt", "Third_Qt", "Fourth_Qt"]
        ].mean()
        ax4.plot(
            ["Q1", "Q2", "Q3", "Q4"],
            quarter_means,
            marker="o",
            label=str(segment)
        )

    ax4.set_xlabel("Quarter")
    ax4.set_ylabel("Average Spend")
    ax4.legend(title=segment_option)
    st.pyplot(fig4)

    st.markdown(
        "Insight: Different segments show distinct seasonal growth patterns.\n\n"
        "Action: Time campaigns differently for each segment."
    )

    # =====================================================
    # CUSTOMER VALUE SEGMENTATION MATRIX
    # =====================================================
    st.subheader("🎯 Customer Value Segmentation Matrix")

    filtered_df['Engagement_Score'] = (
        filtered_df['Site_Visits'] / filtered_df['Site_Visits'].max() * 100
    )

    filtered_df['Spend_Score'] = (
        filtered_df['Total_Spend'] / filtered_df['Total_Spend'].max() * 100
    )

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    colors = filtered_df['Recommend_Binary'].map({1: 'orange', 0: 'blue'})

    ax5.scatter(
        filtered_df['Engagement_Score'],
        filtered_df['Spend_Score'],
        c=colors,
        alpha=0.5,
        s=100
    )

    # Add quadrant lines
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

    # Label quadrants
    ax5.text(25, 75, 'NURTURE\n(Low Visit, High Spend)', 
            ha='center', fontsize=9, style='italic', alpha=0.7)
    ax5.text(75, 75, 'CHAMPIONS\n(High Visit, High Spend)', 
            ha='center', fontsize=9, style='italic', alpha=0.7)
    ax5.text(25, 25, 'AT RISK\n(Low Visit, Low Spend)', 
            ha='center', fontsize=9, style='italic', alpha=0.7)
    ax5.text(75, 25, 'CONVERT\n(High Visit, Low Spend)', 
            ha='center', fontsize=9, style='italic', alpha=0.7)

    ax5.set_xlabel('Engagement Score (Site Visits)', fontsize=11)
    ax5.set_ylabel('Spend Score (Total Spend)', fontsize=11)
    ax5.set_xlim(0, 105)
    ax5.set_ylim(0, 105)

    legend_elements = [
        Patch(facecolor='orange', alpha=0.5, label='Would Recommend'),
        Patch(facecolor='blue', alpha=0.5, label='Would Not Recommend')
    ]
    ax5.legend(handles=legend_elements, loc='upper left')

    st.pyplot(fig5)

    # =====================================================
    # PROFITABILITY HEATMAP WITH SAFE COLUMN SELECTION
    # =====================================================
    st.subheader("🔥 Profitability Heatmap: Gender vs Segment")

    # Ensure index and columns are different
    heatmap_index = 'gender'
    if segment_option == heatmap_index:
        # fallback if segment_option = gender
        heatmap_columns = 'Payment' if 'Payment' in filtered_df.columns else filtered_df.columns[0]
    else:
        heatmap_columns = segment_option

    # Check if columns exist
    if heatmap_index not in filtered_df.columns or heatmap_columns not in filtered_df.columns:
        st.warning("⚠️ Heatmap cannot be generated: required columns missing")
    else:
        heatmap_data = filtered_df.pivot_table(
            values='Profit_Percentage',
            index=heatmap_index,
            columns=heatmap_columns,
            aggfunc='mean'
        )

        fig6, ax6 = plt.subplots(figsize=(10, 5))
        im = ax6.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

        ax6.set_xticks(range(len(heatmap_data.columns)))
        ax6.set_yticks(range(len(heatmap_data.index)))
        ax6.set_xticklabels(heatmap_data.columns)
        ax6.set_yticklabels(heatmap_data.index)

        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not pd.isna(value):
                    ax6.text(j, i, f'{value:.1f}%',
                            ha="center", va="center", 
                            color="black", fontsize=12, weight='bold')

        ax6.set_xlabel(heatmap_columns, fontsize=11)
        ax6.set_ylabel(heatmap_index, fontsize=11)
        cbar = plt.colorbar(im, ax=ax6)
        cbar.set_label('Profit Percentage', rotation=270, labelpad=20)
        plt.tight_layout()
        st.pyplot(fig6)

    # =====================================================
    # QUARTERLY PROGRESSION
    # =====================================================
    st.subheader("📊 Customer Spending Progression Across Quarters")

    quarters = ['First_Qt', 'Second_Qt', 'Third_Qt', 'Fourth_Qt']
    quarterly_avg = filtered_df[quarters].mean()

    retention_rates = []
    for i in range(1, len(quarters)):
        maintained = (filtered_df[quarters[i]] >= filtered_df[quarters[i-1]]).sum()
        total = len(filtered_df)
        retention_rates.append((maintained / total) * 100)

    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT: Quarterly Trend
    ax7a.plot(['Q1', 'Q2', 'Q3', 'Q4'], quarterly_avg, 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax7a.fill_between(range(4), quarterly_avg, alpha=0.3, color='#2E86AB')
    ax7a.set_xlabel('Quarter', fontsize=11)
    ax7a.set_ylabel('Average Spend', fontsize=11)
    ax7a.set_title('Average Quarterly Spend Trend', fontsize=12, weight='bold')
    ax7a.grid(True, alpha=0.3)

    for i in range(1, len(quarterly_avg)):
        pct_change = ((quarterly_avg[i] - quarterly_avg[i-1]) / quarterly_avg[i-1]) * 100
        color = 'green' if pct_change > 0 else 'red'
        ax7a.annotate(f'{pct_change:+.1f}%', xy=(i, quarterly_avg[i]), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, color=color, weight='bold')

    # RIGHT: Retention Rate
    ax7b.bar(['Q1→Q2', 'Q2→Q3', 'Q3→Q4'], retention_rates, 
            color=['#A23B72', '#F18F01', '#C73E1D'])
    ax7b.set_ylabel('% Customers Maintaining/Growing Spend', fontsize=11)
    ax7b.set_title('Quarter-over-Quarter Retention Rate', fontsize=12, weight='bold')
    ax7b.set_ylim(0, 100)

    for i, v in enumerate(retention_rates):
        ax7b.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10, weight='bold')

    plt.tight_layout()
    st.pyplot(fig7)

    st.info(
        "💡 *Key Points*:\n"
        "1. Segment customers strategically for personalized interventions\n"
        "2. Identify profitable vs unprofitable segments for resource allocation\n"
        "3. Track customer lifecycle and predict churn proactively"
    )


    st.subheader("🛍️ Product Category Performance Matrix")

    category_performance = filtered_df.groupby('Last_Purchase').agg({
        'Total_Spend': ['mean', 'sum'],
        'Profit_Percentage': 'mean',
        'Recommend_Binary': 'mean',
        'ID1': 'count'
    }).round(2)

    category_performance.columns = ['Avg_Spend', 'Total_Revenue', 'Avg_Profit', 
                                    'Recommendation_Rate', 'Customer_Count']
    category_performance = category_performance.sort_values('Total_Revenue', ascending=False)

    fig9, ax9 = plt.subplots(figsize=(12, 6))

    # Create bubble chart
    colors_map = {'Food': '#e74c3c', 'Clothes': '#3498db', 
                'Household': '#2ecc71', 'Garden': '#f39c12'}
    colors = [colors_map.get(cat, '#95a5a6') for cat in category_performance.index]

    scatter = ax9.scatter(
        category_performance['Avg_Spend'],
        category_performance['Avg_Profit'],
        s=category_performance['Customer_Count'] * 15,
        c=colors,
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5
    )

    # Add category labels
    for idx, row in category_performance.iterrows():
        ax9.annotate(
            f"{idx}\n({row['Customer_Count']} customers)",
            (row['Avg_Spend'], row['Avg_Profit']),
            fontsize=9,
            ha='center',
            weight='bold'
        )

    ax9.set_xlabel('Average Spend per Customer', fontsize=11)
    ax9.set_ylabel('Average Profit Percentage', fontsize=11)
    ax9.set_title('Category Performance: Spend vs Profitability (bubble size = customer count)', 
                fontsize=12, weight='bold')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig9)



    st.subheader("🔍 Browsing Behavior & Purchase Efficiency")

    browse_conversion = filtered_df.groupby('Browse').agg({
        'Total_Spend': 'mean',
        'Site_Visits': 'mean',
        'Recommend_Binary': 'mean',
        'ID1': 'count'
    }).rename(columns={'ID1': 'Count'})

    browse_conversion['Spend_per_Visit'] = (
        browse_conversion['Total_Spend'] / browse_conversion['Site_Visits']
    )

    # Sort by frequency
    browse_order = ['Once a year', 'Once every few months', 'Once a month', 
                    'More than once a month', 'Once a week', 
                    'More than once a week', 'Daily']
    browse_conversion = browse_conversion.reindex(
        [b for b in browse_order if b in browse_conversion.index]
    )

    fig11, ax11 = plt.subplots(figsize=(12, 6))

    # Create dual-axis visualization
    x_pos = range(len(browse_conversion))
    ax11_twin = ax11.twinx()

    line1 = ax11.plot(x_pos, browse_conversion['Total_Spend'], 
                    'o-', color='#3498db', linewidth=2, markersize=8, 
                    label='Total Spend')
    line2 = ax11_twin.plot(x_pos, browse_conversion['Spend_per_Visit'], 
                        's--', color='#e74c3c', linewidth=2, markersize=8, 
                        label='Spend per Visit')

    ax11.set_xlabel('Browsing Frequency', fontsize=11)
    ax11.set_ylabel('Total Spend', fontsize=11, color='#3498db')
    ax11_twin.set_ylabel('Spend per Visit (Efficiency)', fontsize=11, color='#e74c3c')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(browse_conversion.index, rotation=30, ha='right')
    ax11.tick_params(axis='y', labelcolor='#3498db')
    ax11_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax11.grid(True, alpha=0.3)
    ax11.set_title('Purchase Efficiency vs Browsing Frequency', fontsize=12, weight='bold')

    # Combine legends
    lines1, labels1 = ax11.get_legend_handles_labels()
    lines2, labels2 = ax11_twin.get_legend_handles_labels()
    ax11.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    st.pyplot(fig11)

    st.markdown(
        "**Insight**: Daily browsers have high total spend but lower efficiency - "
        "they may be browsing without clear intent.\n\n"
        "**Action**: Target weekly/monthly browsers with precision campaigns for maximum ROI."
    )


    st.header("💻 Digital Transformation Readiness Assessment")
    st.markdown("*Critical analysis for heritage brand digitalization strategy*")

    # Create digital adoption score
    filtered_df['Digital_Adoption_Score'] = (
        (filtered_df['Site_Visits'] / filtered_df['Site_Visits'].max() * 40) +
        (filtered_df['Browse'].map({
            'Daily': 30, 
            'More than once a week': 25,
            'Once a week': 15,
            'More than once a month': 10,
            'Once a month': 5,
            'Once every few months': 2,
            'Once a year': 1
        }).fillna(0)) +
        (filtered_df['Payment'].map({'Card': 20, 'PayPal': 30}).fillna(15)) +
        (filtered_df['Conven'].map({
            'Extremely convenient': 10,
            'Quite convenient': 7,
            'Neither convenient nor inconvenient': 4,
            'Slightly convenient': 2,
            'Extremely inconvenient': 0
        }).fillna(4))
    )

    # Categorize customers
    def digital_readiness(row):
        score = row['Digital_Adoption_Score']
        spend = row['Total_Spend']
        
        if score >= 70 and spend >= filtered_df['Total_Spend'].quantile(0.75):
            return 'Digital Champions'
        elif score >= 70:
            return 'Digital Natives'
        elif score >= 40 and spend >= filtered_df['Total_Spend'].median():
            return 'Growth Potential'
        elif score >= 40:
            return 'Digital Curious'
        else:
            return 'Traditional Customers'

    filtered_df['Digital_Segment'] = filtered_df.apply(digital_readiness, axis=1)

    fig_digital, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: Scatter plot - Digital Adoption vs Spend
    segment_colors = {
        'Digital Champions': '#27ae60',
        'Digital Natives': '#3498db',
        'Growth Potential': '#f39c12',
        'Digital Curious': '#9b59b6',
        'Traditional Customers': '#95a5a6'
    }

    for segment, color in segment_colors.items():
        segment_data = filtered_df[filtered_df['Digital_Segment'] == segment]
        ax_d1.scatter(segment_data['Digital_Adoption_Score'], 
                    segment_data['Total_Spend'],
                    c=color, label=segment, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    ax_d1.axvline(x=70, color='gray', linestyle='--', alpha=0.5, label='High Digital Adoption')
    ax_d1.axhline(y=filtered_df['Total_Spend'].median(), color='gray', linestyle='--', alpha=0.5, label='Median Spend')

    ax_d1.set_xlabel('Digital Adoption Score', fontsize=12, weight='bold')
    ax_d1.set_ylabel('Total Spend (€)', fontsize=12, weight='bold')
    ax_d1.set_title('Digital Transformation Readiness Matrix', fontsize=13, weight='bold')
    ax_d1.legend(loc='upper left', fontsize=10)
    ax_d1.grid(True, alpha=0.3)

    # RIGHT: Segment Distribution & Potential
    segment_analysis = filtered_df.groupby('Digital_Segment').agg({
        'Total_Spend': 'mean',
        'ID1': 'count',
        'Recommend_Binary': 'mean'
    }).round(2)
    segment_analysis.columns = ['Avg_Spend', 'Count', 'Recommend_Rate']
    segment_analysis = segment_analysis.sort_values('Avg_Spend', ascending=True)

    y_pos = range(len(segment_analysis))
    bars = ax_d2.barh(y_pos, segment_analysis['Avg_Spend'], 
                    color=[segment_colors[seg] for seg in segment_analysis.index], alpha=0.7)

    # Add count annotations
    for i, (idx, row) in enumerate(segment_analysis.iterrows()):
        ax_d2.text(row['Avg_Spend'] + 5, i, 
                f"n={int(row['Count'])} | {row['Recommend_Rate']*100:.0f}% recommend",
                va='center', fontsize=10, weight='bold')

    ax_d2.set_yticks(y_pos)
    ax_d2.set_yticklabels(segment_analysis.index, fontsize=11)
    ax_d2.set_xlabel('Average Spend (€)', fontsize=12, weight='bold')
    ax_d2.set_title('Segment Performance & Size', fontsize=13, weight='bold')
    ax_d2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    st.pyplot(fig_digital)

    st.info(f"""
    **Digital Transformation Priorities:**

    🎯 **Immediate Focus (Digital Champions):** {int(segment_analysis.loc['Digital Champions', 'Count']) if 'Digital Champions' in segment_analysis.index else 0} customers generating €{segment_analysis.loc['Digital Champions', 'Avg_Spend'] if 'Digital Champions' in segment_analysis.index else 0:.0f} avg spend
    - These early adopters are critical for testing new digital initiatives

    🚀 **High Growth Potential:** {int(segment_analysis.loc['Growth Potential', 'Count']) if 'Growth Potential' in segment_analysis.index else 0} customers ready for digital upselling
    - Target with mobile app, personalized recommendations, loyalty programs

    🏛️ **Traditional Customer Bridge:** {int(segment_analysis.loc['Traditional Customers', 'Count']) if 'Traditional Customers' in segment_analysis.index else 0} customers requiring gradual digital onboarding
    - Implement hybrid experiences (online booking for in-store visits)
    - Staff training for digital assistance

    **12-Month Digital Transformation Roadmap:**
    - **Months 1-3:** Digital Champions pilot (AR product visualization, virtual mill tours)
    - **Months 4-6:** Growth Potential expansion (mobile commerce, digital loyalty)
    - **Months 7-9:** Traditional customer bridge programs
    - **Months 10-12:** Full omnichannel integration & measurement
    """)

    st.markdown("---")

    # =============================================================================
    # 3. RURAL HERITAGE MARKET PENETRATION (Foxford Context!)
    # =============================================================================
    st.header("🏘️ Geographic Market Analysis: Irish Rural Strategy")
    st.markdown("*Tailored insights for Foxford's rural Irish heritage market*")

    fig_geo, ((ax_g1, ax_g2), (ax_g3, ax_g4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Analysis by neighbourhood
    neighbourhood_perf = filtered_df.groupby('neighbourhood').agg({
        'Total_Spend': ['mean', 'sum'],
        'Profit_Percentage': 'mean',
        'Site_Visits': 'mean',
        'Recommend_Binary': 'mean',
        'ID1': 'count',
        'Satisfied': lambda x: (x == 'Satisfied').sum() / len(x) * 100
    }).round(2)

    neighbourhood_perf.columns = ['Avg_Spend', 'Total_Revenue', 'Avg_Profit', 
                                'Avg_Visits', 'Recommend_Rate', 'Customer_Count', 'Satisfaction_Rate']

    # TOP LEFT: Market Share by Location
    colors_geo = {'Rural': '#8B4513', 'Suburban': '#3498db', 'Urban': '#95a5a6'}
    wedges, texts, autotexts = ax_g1.pie(neighbourhood_perf['Customer_Count'], 
                                        labels=neighbourhood_perf.index,
                                        autopct='%1.1f%%',
                                        colors=[colors_geo[loc] for loc in neighbourhood_perf.index],
                                        startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    ax_g1.set_title('Customer Distribution by Location', fontsize=13, weight='bold')

    # TOP RIGHT: Profitability by Location
    x_pos = range(len(neighbourhood_perf))
    bars = ax_g2.bar(x_pos, neighbourhood_perf['Avg_Profit'], 
                    color=[colors_geo[loc] for loc in neighbourhood_perf.index], alpha=0.7)

    for i, (idx, val) in enumerate(neighbourhood_perf['Avg_Profit'].items()):
        ax_g2.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=11, weight='bold')

    ax_g2.set_xticks(x_pos)
    ax_g2.set_xticklabels(neighbourhood_perf.index, fontsize=12)
    ax_g2.set_ylabel('Average Profit %', fontsize=12, weight='bold')
    ax_g2.set_title('Profitability by Geographic Segment', fontsize=13, weight='bold')
    ax_g2.grid(True, alpha=0.3, axis='y')

    # BOTTOM LEFT: Digital Engagement by Location
    location_digital = filtered_df.groupby('neighbourhood')['Site_Visits'].mean().sort_values(ascending=False)
    x_pos = range(len(location_digital))
    bars = ax_g3.barh(x_pos, location_digital.values,
                    color=[colors_geo[loc] for loc in location_digital.index], alpha=0.7)

    for i, val in enumerate(location_digital.values):
        ax_g3.text(val + 0.3, i, f'{val:.1f}', va='center', fontsize=11, weight='bold')

    ax_g3.set_yticks(x_pos)
    ax_g3.set_yticklabels(location_digital.index, fontsize=12)
    ax_g3.set_xlabel('Average Site Visits', fontsize=12, weight='bold')
    ax_g3.set_title('Digital Engagement by Location', fontsize=13, weight='bold')
    ax_g3.grid(True, alpha=0.3, axis='x')

    # BOTTOM RIGHT: Spend vs Satisfaction by Location
    for loc in neighbourhood_perf.index:
        loc_data = filtered_df[filtered_df['neighbourhood'] == loc]
        ax_g4.scatter(loc_data['Total_Spend'], 
                    loc_data['Satisfied'].map({'Satisfied': 100, 'Partially satisfied': 50}).fillna(0),
                    c=colors_geo[loc], label=loc, alpha=0.5, s=80)

    ax_g4.set_xlabel('Total Spend (€)', fontsize=12, weight='bold')
    ax_g4.set_ylabel('Satisfaction Score', fontsize=12, weight='bold')
    ax_g4.set_title('Spend vs Satisfaction by Location', fontsize=13, weight='bold')
    ax_g4.legend(fontsize=11)
    ax_g4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_geo)

    # Rural-specific insights for Foxford
    rural_customers = filtered_df[filtered_df['neighbourhood'] == 'Rural']
    rural_avg_spend = rural_customers['Total_Spend'].mean()
    rural_satisfaction = (rural_customers['Satisfied'] == 'Satisfied').sum() / len(rural_customers) * 100

    st.success(f"""
    **Strategic Insights for Foxford's Rural Irish Market:**

    🏡 **Rural Market Performance:**
    - Rural customers: {len(rural_customers)} ({len(rural_customers)/len(filtered_df)*100:.1f}% of base)
    - Average spend: €{rural_avg_spend:.2f} vs Overall: €{filtered_df['Total_Spend'].mean():.2f}
    - Satisfaction rate: {rural_satisfaction:.1f}%

    **Rural-Specific Recommendations for Foxford:**
    1. **Heritage Tourism Integration:** Rural customers may visit as tourists - create "Mill Experience" packages
    2. **Local Community Programs:** Partner with rural craft groups, agricultural fairs
    3. **Click & Collect:** Enable online ordering with mill pickup (reduces delivery barriers)
    4. **Seasonal Rural Campaigns:** Align with agricultural calendar (harvest festivals, Christmas markets)
    5. **Mobile Optimization:** Rural areas often have limited broadband - ensure mobile-first design
    """)

    st.markdown("---")


    st.header("💰 Customer Lifetime Value & Churn Risk Analysis")

    # Calculate CLV proxy
    filtered_df['CLV_Proxy'] = (
        filtered_df['Total_Spend'] * 
        (1 + filtered_df['Recommend_Binary']) *  # Word-of-mouth multiplier
        (filtered_df['Site_Visits'] / 10)  # Engagement factor
    )

    # Churn risk score
    filtered_df['Churn_Risk'] = (
        (filtered_df['Fourth_Qt'] < filtered_df['Third_Qt']).astype(int) * 30 +
        (filtered_df['Satisfied'] == 'Partially satisfied').astype(int) * 40 +
        (filtered_df['Recommend'] == 'No').astype(int) * 30
    )

    def risk_category(score):
        if score >= 70:
            return 'High Risk'
        elif score >= 40:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    filtered_df['Risk_Category'] = filtered_df['Churn_Risk'].apply(risk_category)

    fig_clv, ((ax_c1, ax_c2), (ax_c3, ax_c4)) = plt.subplots(2, 2, figsize=(16, 12))

    # TOP LEFT: CLV Distribution
    ax_c1.hist(filtered_df['CLV_Proxy'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax_c1.axvline(filtered_df['CLV_Proxy'].median(), color='red', linestyle='--', 
                linewidth=2, label=f'Median: €{filtered_df["CLV_Proxy"].median():.0f}')
    ax_c1.set_xlabel('Customer Lifetime Value Proxy (€)', fontsize=12, weight='bold')
    ax_c1.set_ylabel('Number of Customers', fontsize=12, weight='bold')
    ax_c1.set_title('CLV Distribution', fontsize=13, weight='bold')
    ax_c1.legend(fontsize=11)
    ax_c1.grid(True, alpha=0.3)

    # TOP RIGHT: Churn Risk Distribution
    risk_counts = filtered_df['Risk_Category'].value_counts()
    colors_risk = {'Low Risk': '#27ae60', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
    bars = ax_c2.bar(range(len(risk_counts)), risk_counts.values,
                    color=[colors_risk[cat] for cat in risk_counts.index], alpha=0.7)

    for i, (cat, val) in enumerate(risk_counts.items()):
        ax_c2.text(i, val + 5, f'{val}\n({val/len(filtered_df)*100:.1f}%)', 
                ha='center', fontsize=11, weight='bold')

    ax_c2.set_xticks(range(len(risk_counts)))
    ax_c2.set_xticklabels(risk_counts.index, fontsize=12)
    ax_c2.set_ylabel('Number of Customers', fontsize=12, weight='bold')
    ax_c2.set_title('Churn Risk Distribution', fontsize=13, weight='bold')
    ax_c2.grid(True, alpha=0.3, axis='y')

    # BOTTOM LEFT: CLV vs Churn Risk Scatter
    for risk, color in colors_risk.items():
        risk_data = filtered_df[filtered_df['Risk_Category'] == risk]
        ax_c3.scatter(risk_data['CLV_Proxy'], risk_data['Churn_Risk'],
                    c=color, label=risk, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

    ax_c3.set_xlabel('Customer Lifetime Value (€)', fontsize=12, weight='bold')
    ax_c3.set_ylabel('Churn Risk Score', fontsize=12, weight='bold')
    ax_c3.set_title('CLV vs Churn Risk Matrix', fontsize=13, weight='bold')
    ax_c3.legend(fontsize=11)
    ax_c3.grid(True, alpha=0.3)

    # BOTTOM RIGHT: High-Value At-Risk Customers
    high_value_at_risk = filtered_df[
        (filtered_df['CLV_Proxy'] >= filtered_df['CLV_Proxy'].quantile(0.75)) &
        (filtered_df['Risk_Category'] == 'High Risk')
    ]

    retention_priorities = filtered_df.groupby('Risk_Category').agg({
        'CLV_Proxy': ['mean', 'sum'],
        'ID1': 'count'
    }).round(0)
    retention_priorities.columns = ['Avg_CLV', 'Total_CLV', 'Count']
    retention_priorities = retention_priorities.sort_values('Total_CLV', ascending=False)

    y_pos = range(len(retention_priorities))
    bars = ax_c4.barh(y_pos, retention_priorities['Total_CLV'],
                    color=[colors_risk[cat] for cat in retention_priorities.index], alpha=0.7)

    for i, (idx, row) in enumerate(retention_priorities.iterrows()):
        ax_c4.text(row['Total_CLV'] + 1000, i,
                f"€{row['Total_CLV']:,.0f}\n({int(row['Count'])} customers)",
                va='center', fontsize=10, weight='bold')

    ax_c4.set_yticks(y_pos)
    ax_c4.set_yticklabels(retention_priorities.index, fontsize=12)
    ax_c4.set_xlabel('Total CLV at Risk (€)', fontsize=12, weight='bold')
    ax_c4.set_title('Revenue at Risk by Segment', fontsize=13, weight='bold')
    ax_c4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    st.pyplot(fig_clv)

    st.error(f"""
    ⚠️ **URGENT: High-Value Customer Retention Alert**

    **{len(high_value_at_risk)} high-value customers at HIGH risk** of churn
    - Potential revenue loss: **€{high_value_at_risk['CLV_Proxy'].sum():,.0f}**
    - Average CLV: €{high_value_at_risk['CLV_Proxy'].mean():.0f} per customer

    **Immediate Retention Actions:**
    1. **Personal Outreach:** Direct contact from senior management within 48 hours
    2. **VIP Program:** Exclusive mill tours, early access to new collections
    3. **Win-back Offer:** 20% discount + free shipping on next purchase
    4. **Feedback Loop:** Survey to understand dissatisfaction drivers
    5. **Loyalty Program:** Points system with milestone rewards

    **12-Month Retention Strategy:**
    - Target: Reduce high-risk segment from {risk_counts['High Risk']/len(filtered_df)*100:.1f}% to <10%
    - Investment: €50,000 retention budget (ROI potential: €{high_value_at_risk['CLV_Proxy'].sum()*0.5:,.0f})
    """)

    st.markdown("---")



else:
    # =====================================================
    # ML PREDICTIONS TAB
    # =====================================================
    st.title("🤖 Machine Learning Predictions")
    st.markdown("*Predict customer behavior using advanced ML models*")
    
    # =====================================================
    # 1️⃣ 3D-STYLE BUBBLE CHART: SPEND vs VISITS vs PROFIT
    # =====================================================
    st.subheader("💎 Revenue Opportunity Matrix: Spend × Engagement × Profitability")

    col_left1, col_right1 = st.columns([2, 1])

    with col_left1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Create scatter with bubble size and color
        scatter = ax1.scatter(
            filtered_df["Site_Visits"],
            filtered_df["Total_Spend"],
            s=filtered_df["Profit_Percentage"] * 10,  # Size by profit
            c=filtered_df["Profit_Percentage"],
            cmap='RdYlGn',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add regression line
        from sklearn.linear_model import LinearRegression
        X = filtered_df[["Site_Visits"]]
        y = filtered_df["Total_Spend"]
        model = LinearRegression()
        model.fit(X, y)
        x_line = np.linspace(X.min().values[0], X.max().values[0], 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax1.plot(x_line, y_line, 'r--', linewidth=2.5, label=f'Trend: +€{model.coef_[0]:.2f} per visit', alpha=0.8)
        
        # Add quadrant lines
        median_visits = filtered_df["Site_Visits"].median()
        median_spend = filtered_df["Total_Spend"].median()
        
        ax1.axvline(x=median_visits, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax1.axhline(y=median_spend, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Quadrant labels
        ax1.text(filtered_df["Site_Visits"].max() * 0.85, 
                 filtered_df["Total_Spend"].max() * 0.95,
                 'HIGH VALUE\nHIGH ENGAGE', fontsize=9, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax1.text(filtered_df["Site_Visits"].min() * 1.1,
                 filtered_df["Total_Spend"].max() * 0.95,
                 'HIGH VALUE\nLOW ENGAGE', fontsize=9, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
        
        ax1.text(filtered_df["Site_Visits"].max() * 0.85,
                 filtered_df["Total_Spend"].min() * 1.1,
                 'LOW VALUE\nHIGH ENGAGE', fontsize=9, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax1.set_xlabel("Site Visits (Engagement)", fontsize=11, weight='bold')
        ax1.set_ylabel("Total Spend (€)", fontsize=11, weight='bold')
        ax1.set_title('Strategic Positioning: Bubble Size = Profit Margin', 
                      fontsize=12, weight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Profit %', rotation=270, labelpad=15, weight='bold')
        
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_right1:
        st.markdown(f"""
        **📊 Key Insights:**
        
        **Regression Model:**
        - Coefficient: €{model.coef_[0]:.2f}
        - Each visit = €{model.coef_[0]:.2f} more spend
        - R² Score: {model.score(X, y):.3f}
        
        **Strategic Zones:**
        - 🟢 **Top Right:** Your stars - retain!
        - 🟡 **Top Left:** High spenders with low engagement - upsell opportunity
        - 🔴 **Bottom Right:** Over-engaged, under-converting - optimize journey
        
        **Action:** Focus on moving customers upward and rightward on this matrix.
        """)

    st.markdown("---")

    # =====================================================
    # 2️⃣ ADVANCED CLUSTERING WITH PROFILES
    # =====================================================
    st.subheader("🎯 Customer Segmentation with Behavioral Profiles")

    col_left2, col_right2 = st.columns([1, 2])

    with col_left2:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        k = st.slider("Number of Clusters", 2, 6, 3)
        
        cluster_features = filtered_df[
            ["Total_Spend", "Site_Visits", "Profit_Percentage"]
        ]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_features)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        filtered_df["Cluster"] = kmeans.fit_predict(scaled)
        
        # Cluster summary
        st.markdown("**Cluster Summary:**")
        for i in range(k):
            cluster_data = filtered_df[filtered_df["Cluster"] == i]
            st.markdown(f"""
            **Cluster {i}** (n={len(cluster_data)})
            - Avg Spend: €{cluster_data['Total_Spend'].mean():.0f}
            - Avg Visits: {cluster_data['Site_Visits'].mean():.1f}
            - Avg Profit: {cluster_data['Profit_Percentage'].mean():.1f}%
            """)

    with col_right2:
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Cluster scatter with enhanced visuals
        colors = plt.cm.Set3(np.linspace(0, 1, k))
        
        for i in range(k):
            cluster_data = filtered_df[filtered_df["Cluster"] == i]
            ax2a.scatter(
                cluster_data["Site_Visits"],
                cluster_data["Total_Spend"],
                c=[colors[i]],
                label=f'Cluster {i}',
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add cluster centers
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        ax2a.scatter(
            centers_original[:, 1],  # Site_Visits
            centers_original[:, 0],  # Total_Spend
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=5
        )
        
        ax2a.set_xlabel("Site Visits", fontsize=11, weight='bold')
        ax2a.set_ylabel("Total Spend (€)", fontsize=11, weight='bold')
        ax2a.set_title('Cluster Distribution', fontsize=12, weight='bold')
        ax2a.legend(loc='best', framealpha=0.9)
        ax2a.grid(True, alpha=0.3)
        
        # Right: Cluster comparison bars
        cluster_summary = filtered_df.groupby("Cluster").agg({
            'Total_Spend': 'mean',
            'Site_Visits': 'mean',
            'Profit_Percentage': 'mean'
        }).reset_index()
        
        x = np.arange(k)
        width = 0.25
        
        ax2b.bar(x - width, cluster_summary['Total_Spend']/10, width, 
                 label='Spend (÷10)', color='#1f77b4', alpha=0.7)
        ax2b.bar(x, cluster_summary['Site_Visits'], width, 
                 label='Visits', color='#ff7f0e', alpha=0.7)
        ax2b.bar(x + width, cluster_summary['Profit_Percentage'], width, 
                 label='Profit %', color='#2ca02c', alpha=0.7)
        
        ax2b.set_xlabel('Cluster', fontsize=11, weight='bold')
        ax2b.set_ylabel('Value', fontsize=11, weight='bold')
        ax2b.set_title('Cluster Characteristics', fontsize=12, weight='bold')
        ax2b.set_xticks(x)
        ax2b.set_xticklabels([f'C{i}' for i in range(k)])
        ax2b.legend()
        ax2b.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # =====================================================
    # 3️⃣ INTERACTIVE SCENARIO DASHBOARD
    # =====================================================
    st.subheader("🧪 Predictive Scenario Modeling")

    scenario_visits = st.slider(
        "Scenario: Avg Site Visits",
        1, int(filtered_df["Site_Visits"].max()), 10
    )

    scenario_satisfaction = st.slider(
        "Scenario: Satisfaction Boost (%)",
        0, 20, 5
    )

    col_left3, col_right3 = st.columns([1, 2])

    with col_left3:
        avg_spend = filtered_df["Total_Spend"].mean()
        rec_rate = filtered_df["Recommend_Binary"].mean()
        
        predicted_spend = model.predict([[scenario_visits]])[0]
        simulated_rec_rate = min(rec_rate * (1 + scenario_satisfaction / 100), 1)
        
        # Calculate revenue impact
        current_revenue = len(filtered_df) * avg_spend
        projected_revenue = len(filtered_df) * predicted_spend
        revenue_change = projected_revenue - current_revenue
        
        st.markdown(f"""
        ### 📈 Current State
        - Avg Spend: **€{avg_spend:.2f}**
        - Rec Rate: **{rec_rate*100:.1f}%**
        - Total Revenue: **€{current_revenue:,.0f}**
        
        ### 🚀 Projected State
        - Avg Spend: **€{predicted_spend:.2f}**
        - Rec Rate: **{simulated_rec_rate*100:.1f}%**
        - Total Revenue: **€{projected_revenue:,.0f}**
        
        ### 💰 Impact
        - Revenue Change: **€{revenue_change:+,.0f}**
        - Change %: **{(revenue_change/current_revenue*100):+.1f}%**
        """)

    with col_right3:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Revenue comparison
        categories = ['Current', 'Projected']
        revenues = [current_revenue, projected_revenue]
        colors_bar = ['#3498db', '#2ecc71' if revenue_change > 0 else '#e74c3c']
        
        bars = ax3a.bar(categories, revenues, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, rev in zip(bars, revenues):
            height = bar.get_height()
            ax3a.text(bar.get_x() + bar.get_width()/2., height,
                     f'€{rev:,.0f}',
                     ha='center', va='bottom', fontsize=12, weight='bold')
        
        ax3a.set_ylabel('Total Revenue (€)', fontsize=11, weight='bold')
        ax3a.set_title('Revenue Projection', fontsize=12, weight='bold')
        ax3a.grid(axis='y', alpha=0.3)
        
        # Right: Metrics gauge
        metrics = ['Avg Spend', 'Rec Rate', 'Revenue']
        current_vals = [avg_spend, rec_rate*100, current_revenue/1000]
        projected_vals = [predicted_spend, simulated_rec_rate*100, projected_revenue/1000]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax3b.barh(x_pos - width/2, current_vals, width, label='Current', 
                  color='#95a5a6', alpha=0.7)
        ax3b.barh(x_pos + width/2, projected_vals, width, label='Projected',
                  color='#27ae60', alpha=0.7)
        
        ax3b.set_yticks(x_pos)
        ax3b.set_yticklabels(metrics)
        ax3b.set_xlabel('Value', fontsize=11, weight='bold')
        ax3b.set_title('Metric Comparison', fontsize=12, weight='bold')
        ax3b.legend()
        ax3b.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("---")

    # =====================================================
    # 4️⃣ QUARTERLY TREND (ENHANCED WITH SEGMENTS)
    # =====================================================
    st.subheader(f"📆 Quarterly Revenue Trends by {segment_option}")

    fig4, ax4 = plt.subplots(figsize=(14, 6))

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    qt_cols = ["First_Qt", "Second_Qt", "Third_Qt", "Fourth_Qt"]

    # Get segments
    segments = filtered_df[segment_option].unique()
    colors_palette = plt.cm.tab10(np.linspace(0, 1, len(segments)))

    for idx, segment in enumerate(segments):
        segment_data = filtered_df[filtered_df[segment_option] == segment]
        quarter_means = segment_data[qt_cols].mean()
        
        ax4.plot(quarters, quarter_means, 
                 marker='o', 
                 linewidth=2.5,
                 markersize=10,
                 label=f'{segment} (n={len(segment_data)})',
                 color=colors_palette[idx])

    # Add overall trend
    overall_means = filtered_df[qt_cols].mean()
    ax4.plot(quarters, overall_means, 
             marker='s', 
             linewidth=3,
             markersize=8,
             label='Overall Average',
             color='black',
             linestyle='--',
             alpha=0.7)

    ax4.set_xlabel("Quarter", fontsize=12, weight='bold')
    ax4.set_ylabel("Average Spend (€)", fontsize=12, weight='bold')
    ax4.set_title(f'Seasonal Performance Analysis by {segment_option}', 
                  fontsize=13, weight='bold', pad=15)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(bottom=0)

    plt.tight_layout()
    st.pyplot(fig4)

    st.info(f"💡 **Insight:** Track which {segment_option} segments show growth vs. decline patterns. "
            f"Use this to time campaigns and allocate resources strategically.")

    st.markdown("---")

    # =====================================================
    # 5️⃣ EXECUTIVE SUMMARY EXPORT
    # =====================================================
    st.subheader("📤 Executive Summary Export")

    col_export1, col_export2 = st.columns([2, 1])

    with col_export1:
        summary_df = pd.DataFrame({
            "Metric": [
                "Avg Spend",
                "Recommendation Rate",
                "Avg Profit Margin",
                "Regression Coefficient (Visits → Spend)",
                "Best Cluster Avg Spend",
                "Projected Spend (Scenario)",
                "Revenue Impact (Scenario)"
            ],
            "Value": [
                f"€{avg_spend:.2f}",
                f"{rec_rate * 100:.1f}%",
                f"{filtered_df['Profit_Percentage'].mean():.1f}%",
                f"€{model.coef_[0]:.2f}",
                f"€{filtered_df.groupby('Cluster')['Total_Spend'].mean().max():.2f}",
                f"€{predicted_spend:.2f}",
                f"€{revenue_change:+,.0f}"
            ]
        })
        
        st.dataframe(summary_df, use_container_width=True)

    with col_export2:
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Summary CSV",
            csv,
            "executive_summary.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Also offer cluster data
        cluster_csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Dataset CSV",
            cluster_csv,
            "customer_data_with_clusters.csv",
            "text/csv",
            use_container_width=True
        )

    # -----------------------------
    # Final Insight
    # -----------------------------
    st.success(
        "✅ This platform integrates predictive analytics, segmentation intelligence, "
        "scenario modeling, and actionable trend analysis for data-driven decision making."
    )