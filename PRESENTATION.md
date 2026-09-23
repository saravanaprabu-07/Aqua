# AquaTrend — Hackathon Presentation Script
*(~4.5 minutes total, for a 3-member team)*

## 1. Problem — 30 seconds
"Groundwater levels across India are falling in many regions, but decisions about where to
focus monitoring and recharge efforts are hard to make without knowing where the decline is
real, persistent, and worsening — versus normal seasonal ups and downs. AquaTrend uses machine
learning to forecast future groundwater levels and flag locations with a genuine, sustained
declining trend."

## 2. Dataset — 30 seconds
"We used real quarterly groundwater-level data from India's Central Ground Water Board,
covering 60 districts from 1996 to 2017 — about 5,000 real measurements, no synthetic data.
The key variable is 'depth to water level' — how many metres below the ground the water table
sits."

## 3. Preprocessing — 30 seconds
"We cleaned the data — removing duplicates and impossible values — then engineered time-series
features: lag values from 1, 2 and 4 quarters back, a rolling yearly average, and
season/month/year features, since groundwater is strongly seasonal."

## 4. ML Algorithm — 45 seconds
"We trained a Random Forest Regressor to predict next-quarter groundwater depth using those lag
and seasonal features. Critically, we split the data chronologically — training only on the
past and testing only on the most recent two years — never randomly, because that would leak
future information into training. For multi-step forecasts, we predict one quarter, feed that
prediction back in, and repeat."

## 5. Results — 45 seconds
"On the held-out chronological test set, our model achieves a Mean Absolute Error of about
0.49 metres and an R² of 0.91 — meaning it explains around 91% of the variation in real
groundwater depth changes. We also built a data-driven trend classifier: using the actual
distribution of trend slopes across our 60 districts, we classify each as Improving, Stable,
Mild Decline, or Significant Decline. Punjab districts like Ludhiana and Sangrur came out as
'Significant Decline / High Priority' — which matches what's widely reported about groundwater
stress in that region."

## 6. Dashboard — 45 seconds
"Everything comes together in our AquaTrend Streamlit dashboard: pick any district to see its
historical groundwater chart alongside the ML forecast, clearly separated by color and a marked
boundary line. There's a trend-analysis tab with monitoring-priority recommendations, a seasonal
analysis tab, and a region-comparison tab to compare multiple districts side by side."

## 7. Innovation — 30 seconds
"Rather than bolt on a generic time-series library, we built an explainable, data-driven
classification system — our decline thresholds come directly from this dataset's own statistics,
not arbitrary numbers — and we clearly separate what's measured from what's predicted throughout
the whole platform, which we think is essential for a tool meant to support real monitoring
decisions."

## 8. Future Scope — 30 seconds
"With more time, we'd bring in rainfall and extraction data to improve forecast accuracy,
extend the dataset closer to the present day, and move from district-level to individual
well-level forecasting for more precise, local warnings."

---

## Possible Judge Questions & Simple Answers

**Q: Why Random Forest and not LSTM/ARIMA?**
A: Random Forest is simple to explain, works well with a moderate amount of tabular
lag-feature data, and doesn't need the large amount of data that deep learning models like
LSTMs typically require. It was the right tool for the size and structure of our dataset.

**Q: How do you know your train/test split doesn't leak future data?**
A: For every district, we always used the earliest quarters for training and the most recent 8
quarters (2 years) purely for testing — never a random shuffle. The test set is always strictly
later in time than the training set.

**Q: Why didn't you include rainfall?**
A: The dataset we found and verified didn't include rainfall for these wells. Per the hackathon
rules, we chose not to fabricate that data rather than invent numbers — we relied only on the
genuine groundwater time-series itself.

**Q: How were your decline-category thresholds decided?**
A: They're computed directly from the mean and standard deviation of the actual trend slopes
across our 60 districts — not picked arbitrarily. That's documented in `trend_analysis.py`.

**Q: Can this system stop groundwater depletion?**
A: No — it's a monitoring and decision-support tool. It flags where attention (monitoring,
review of extraction/recharge practices) is needed; it doesn't control extraction directly.

**Q: What does "depth to water level" actually mean, in plain terms?**
A: It's how far down you'd have to dig or drill to reach water, measured in metres. A bigger
number is worse (water is deeper down); a smaller number is better (water is closer to the
surface).
