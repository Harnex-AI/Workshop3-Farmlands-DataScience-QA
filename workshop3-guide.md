# Data Science & QA Workshop

## Overview
This workshop provides **AI-assisted development prompts** and guidance for building, testing, and validating data science workflows for Farmlands Co-operative in New Zealand.

Participants will:
- Implement data cleaning, exploration, and modeling modules.
- Create unit and integration tests.
- Use AI tools like Cursor AI to assist in development.
- Focus on business value and agricultural context.

---

## 🧹 Data Cleaning Module

**File:** `src/farmlands_analytics/data_cleaning/cleaner.py`

### 1. Handle Missing Values (`handle_missing_values`)
**Goal:** Intelligent missing value handling based on data type and missing percentage.

**Acceptance Criteria:**
- Strategy based on:
  - Column data type (numeric, categorical, datetime)
  - Missing value percentage (<5%, 5–20%, >20%)
  - Agricultural business logic
- Numerical: mean, median, forward fill
- Categorical: mode, forward fill, "Unknown"
- Dates: interpolation or seasonal logic
- High missing %: flag for review or drop
- Log all actions; return cleaned DataFrame + summary

---

### 2. Standardize Product Names (`standardize_product_names`)
**Goal:** Clean and group similar agricultural product names.

**Acceptance Criteria:**
- Normalize case, spacing, punctuation
- Map common variations (e.g., “46%” vs “46 percent” vs “46-0-0”)
- Preserve original in separate column
- Extensible mapping dictionary

---

### 3. Validate Farm IDs (`validate_farm_ids`)
**Goal:** Enforce consistent `F###` format.

**Acceptance Criteria:**
- Regex validation (F001–F999)
- Suggest corrections
- Flag invalid/null
- Return cleaned DataFrame + issues list

---

### 4. Clean Price Outliers (`clean_price_outliers`)
**Goal:** Detect and handle outliers per product category.

**Acceptance Criteria:**
- Methods: IQR, Z-score, Isolation Forest
- Product-aware thresholds
- Cap, remove, or mark for review
- Log detections + before/after stats

---

### 5. Standardize Date Formats (`standardize_date_formats`)
**Goal:** Parse multiple date formats, create derived fields.

**Acceptance Criteria:**
- Output in ISO (YYYY-MM-DD)
- Features: season, quarter, month, year, weekday, weekend flag
- Flag future or invalid dates
- Handle partial dates

---

## 🔍 Data Exploration Module

**File:** `src/farmlands_analytics/data_exploration/explorer.py`

1. **Generate Summary Statistics** (`generate_summary_statistics`)
2. **Analyze Seasonal Patterns** (`analyze_seasonal_patterns`)
3. **Explore Regional Differences** (`explore_regional_differences`)
4. **Product Performance Analysis** (`product_performance_analysis`)
5. **Correlation Analysis** (`correlation_analysis`)

**General Acceptance Criteria:**
- Statistical summaries (numeric + categorical)
- Agricultural insights (seasonal peaks, regional variations)
- Visualisations: histograms, heatmaps, decomposition plots
- Actionable recommendations

---

## 🤖 Machine Learning Modeling Module

**File:** `src/farmlands_analytics/modeling/predictor.py`

1. **Demand Forecasting Model** (`demand_forecasting_model`)
2. **Price Prediction Model** (`price_prediction_model`)
3. **Customer Segmentation Model** (`customer_segmentation_model`)
4. **Weather Impact Prediction** (`weather_impact_prediction`)

**General Acceptance Criteria:**
- Use multiple algorithms (statistical, ML, DL)
- Feature engineering with weather, seasonality, economic indicators
- Agricultural domain knowledge integration
- Model validation + interpretability

---

## ✅ Unit Testing (QA)

**Files:**  
`tests/unit/test_data_cleaning.py`  
`tests/unit/test_data_exploration.py`

**Acceptance Criteria:**
- Happy path + edge case tests
- Agricultural-specific tests (NZ seasons, farm IDs, product names)
- Performance checks
- Mocks for external dependencies

---

## 🔄 Integration Testing (QA)

**File:** `tests/integration/test_end_to_end_pipeline.py`

**Acceptance Criteria:**
- Full pipeline flow from raw → cleaned → explored → modeled
- Multi-region, multi-season scenarios
- Performance monitoring
- Production readiness checks

---

## 💡 AI Prompting Tips

1. Always provide **business context** (NZ agriculture)
2. Specify **data characteristics**
3. Include **domain knowledge** (seasons, crop cycles)
4. Request **error handling**, **docstrings**, and **examples**
5. Aim for **actionable insights**

---

## 🚀 Workshop Strategy

- **Start Simple:** Build core functionality first
- **Add Features:** Extend with advanced techniques
- **Optimise:** Improve performance and edge case handling
- **Integrate:** Connect with other modules
- **Validate:** Comprehensive testing and stakeholder review

---

## 🎯 Definition of Done
- All TODOs implemented with tests passing
- Business relevance validated
- Documentation + logging complete
- Models and insights actionable for Farmlands management