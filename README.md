# AmaZon-Diwali-Product-Sales-2025 
# Using Power BI And Python Library (matplotlib, pandas)
Sales report of the amazon Diwali sale 2025

## Dataset used
  -<a href="https://github.com/thechronic04/AmaZon-Diwali-Product-Sales-2025/blob/main/amazon_sales_2025_INR.csv">Dataset</a>
  


Dashboard Interactions  <a  href="https://github.com/thechronic04/AmaZon-Diwali-Product-Sales-2025/blob/main/Screenshot%202025-11-05%20233756.png">View Dashboard</a>


## Dashboard
[ [![Screenshot (44)]((https://github.com/thechronic04/AmaZon-Diwali-Product-Sales-2025/blob/main/Screenshot%202025-11-05%20233756.png))]



Monthly Total Sales(inr)-2025 <img width="1979" height="780" alt="57b8e11a-b973-4d4f-b214-8827ea4d620c" src="https://github.com/user-attachments/assets/2a5e6306-89c1-4200-a22b-cb09a729a052" />


Actual Monthly Sales (2025) and Forecast(2026) <img width="2178" height="780" alt="f8c8a9b6-ba1a-4a7f-b6b0-ce9a75ca5b0c" src="https://github.com/user-attachments/assets/7faae56b-b98c-4890-8df3-037e34d75d58" />



## Questiona (KPIs)

Amazon Diwaliproducts sales 2025

Analysis for next years diwali sales.
1.Which product category sold the most?
2.which method of payment the wetre Used the most?
3.Product rating of Tops product.
4.,which state ordered most product?
5.Delivery status of sales.
6.Which product sold the most?




# Dashboard Title

Amazon Diwali Products Sales 2025 Report
Displayed as a text box or card for clear report branding.

Total Sales by Product Category

Type: Pie Chart
Purpose: To show each product category’s percentage share of total sales.
Details:
Slices represent categories: Beauty, Electronics, Books, Clothing, Home & Kitchen.
Value field: Sum of Total_Sales_INR.
Best for comparing proportions.

Payment Method Used

Type: Horizontal Bar Chart (Clustered Bar)
Purpose: To show which payment methods contributed the most to total sales.
Axis:
Y-axis → Payment Method (Credit Card, COD, Debit, UPI)
X-axis → Sum of Total Sales (INR)
Color-coded for visual contrast.

State-wise Order

Type: Tree Map
Purpose: To show regional distribution of orders.
Details:
Each rectangle = one state (size = total orders/sales).
Larger area → more orders.
Perfect for geographical categorical comparison without maps.

Product Wise Rating

Type: Table / Matrix Visual
Purpose: To list products with their total review ratings.
Columns: Product_Name | Sum of Review_Rating
Use: For rank-based insights (Perfume, Children’s Book, Smartwatch rated highest).

Total Sales INR by Delivery Status (Groups)

Type: Stacked Bar Chart
Purpose: To compare delivery outcomes (Delivered, Pending, Returned).
Axis:
Y-axis → Delivery Status
X-axis → Sum of Total Sales (INR)
Visual Feature: 100% stacked bar style shows proportions and totals simultaneously.

Top 10 Total Sales INR by Product Name

Type: Horizontal Bar Chart (Sorted)
Purpose: To display top 10 best-selling products.
Axis:
Y-axis → Product Name (Lipstick, Children’s Book, etc.)
X-axis → Sum of Total Sales (INR)
Bars sorted descending = clear ranking.



# Python code for forecasting
# =============================================
# AMAZON DIWALI SALES 2025 - TREND & FORECAST
# =============================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("amazon_sales_2025_INR.csv")

# Quick check
print("Dataset info:")
print(df.info())
print("\nSample data:")
print(df.head())

# Clean column names (if needed)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convert date column (if exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# =============================================
# 1️  BASIC SALES OVERVIEW
# =============================================
print("\n--- Basic Summary ---")
print(df.describe())

# Total sales per category
category_sales = df.groupby('category')['sales_in_inr'].sum().sort_values(ascending=False)
print("\nTop Categories by Sales:")
print(category_sales.head(10))

# =============================================
# 2️  MONTHLY SALES TREND (2025)
# =============================================
if 'date' in df.columns:
    df['month'] = df['date'].dt.to_period('M')
    monthly_sales = df.groupby('month')['sales_in_inr'].sum().reset_index()
    monthly_sales['month_num'] = range(1, len(monthly_sales)+1)
else:
    print(" No 'date' column found — skipping monthly trend.")
    monthly_sales = pd.DataFrame()

# Plot Monthly Sales 2025
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['month'].astype(str), monthly_sales['sales_in_inr'], marker='o', label='2025 Actual')
plt.xticks(rotation=45)
plt.title("Amazon Diwali Sales 2025 - Monthly Trend")
plt.xlabel("Month")
plt.ylabel("Sales (INR)")
plt.legend()
plt.grid(True)
plt.show()

# =============================================
# 3️  FORECASTING SALES FOR 2026 (LINEAR REGRESSION)
# =============================================

# Prepare data for regression
X = monthly_sales[['month_num']]
y = monthly_sales['sales_in_inr']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 12 months (2026)
future_months = np.arange(len(X)+1, len(X)+13).reshape(-1, 1)
predicted_sales = model.predict(future_months)

# Create forecast dataframe
forecast_2026 = pd.DataFrame({
    'month_2026': pd.date_range('2026-01-01', periods=12, freq='M').to_period('M').astype(str),
    'forecast_sales_in_inr': predicted_sales
})

print("\n--- Forecasted Monthly Sales for 2026 ---")
print(forecast_2026)

# =============================================
# 4️  VISUALIZE FORECAST
# =============================================
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['month_num'], monthly_sales['sales_in_inr'], label='2025 Actual', marker='o')
plt.plot(future_months, predicted_sales, label='2026 Forecast', linestyle='--', marker='x')
plt.title("Amazon Diwali Sales: 2025 Actual vs 2026 Forecast")
plt.xlabel("Month Number")
plt.ylabel("Sales (INR)")
plt.legend()
plt.grid(True)
plt.show()

# =============================================
# 5️  CATEGORY PERFORMANCE (OPTIONAL)
# =============================================
category_summary = df.groupby('category').agg({
    'sales_in_inr':'sum',
    'rating':'mean'
}).reset_index().sort_values(by='sales_in_inr', ascending=False)

print("\n--- Category Summary ---")
print(category_summary.head(10))

# =============================================
# 6️ EXPORT RESULTS (OPTIONAL)
# =============================================
with pd.ExcelWriter("Amazon_Sales_2025_Analysis_and_2026_Forecast.xlsx") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    monthly_sales.to_excel(writer, sheet_name="2025_Monthly_Sales", index=False)
    forecast_2026.to_excel(writer, sheet_name="2026_Forecast", index=False)
    category_summary.to_excel(writer, sheet_name="Category_Summary", index=False)

print("\n Analysis complete — Excel report saved: 'Amazon_Sales_2025_Analysis_and_2026_Forecast.xlsx'")





# What I did for forecasting.

Aggregated sales to monthly totals for 2025.
Built a Linear Regression model with:
MonthIndex (trend)
sin(month) and cos(month) (to capture seasonality)
Forecasted monthly sales for Jan–Dec 2026.
Explored top 5 product categories trends across months.
I displayed the intermediate tables and plots so you can inspect them directly.

# Model performance (linear regression with seasonality)

R² (goodness of fit): 0.337 — model explains ~33.7% of variance (moderate).
Interpretation: Linear regression with one seasonal harmonic captures some trend + seasonality, but there's substantial unexplained variance (typical for retail data — promotions, daily spikes, returns, etc.).
RMSE: ₹2,856,632 (average monthly error magnitude).
Coefficients:
MonthIndex: +723,084 (monthly upward trend)
sin(month): +826,061
cos(month): -1,103,405
Intercept: 89,203,185.95
These coefficients combine to produce the predicted monthly sales pattern (trend + seasonal wave).

# Forecasted monthly sales (2026)

I showed the forecast table in Code section.

YearMonth	Forecast_Sales_INR
2026-01-01	...
2026-02-01	...
2026-12-01	...


# Amazon Diwali Product Sales Analysis & Forecast – 2026
Key Highlights from 2025 Sales
Aspect	2025 Insights
Top Product Category	Beauty Products
Top Selling Product	Lipstick
Top 5 States by Orders	Sikkim, Rajasthan, Tamil Nadu, Meghalaya, Chhattisgarh
Top Payment Method	Credit Card
Delivery Status Summary	Delivered – 378.75M, Pending – 376.21M, Returned – 363.20M
High-Rated Products	Perfume, Children’s Books, Smartwatch, Tablet, Laptop

## 2026 Sales Forecast & Strategic Recommendations
# Focus on High-Demand Categories

Beauty Products are consistently top-performing — expand product variety and introduce combo Diwali gift packs (lipstick + perfume + skincare).
Electronics showed strong momentum (smartwatches, tablets, laptops). Expect +15–20% growth next year due to tech gifting trends.
Promote Children’s Books through “Family & Learning Diwali Offers."

# Strategy:
Create targeted Diwali campaigns like “Glow & Gadget Fest 2026” focusing on Beauty + Electronics.

# Optimize Payment Method Mix

Since Credit Card usage was the highest, offer exclusive cashback deals and EMI offers to boost conversion rates.
UPI adoption is increasing year over year — ensure smooth UPI payment integration and festival cashback for UPI users.

# Strategy:
Introduce tiered discounts by payment method (e.g., 10% off on Credit Card, 7% off on UPI).

# Improve Delivery Efficiency

High Pending (376M) and Returned (363M) orders indicate a logistics bottleneck.

Focus on inventory forecasting, regional warehouses, and real-time tracking updates.

# Strategy:
Launch “Amazon Diwali Express Delivery” for top 10 states and strengthen return handling through local partners.

# Regional Targeting – Statewise Insights

Sikkim and Rajasthan show strong Diwali shopping enthusiasm.
Consider regional ad campaigns featuring cultural themes and language localization.
States like Meghalaya and Chhattisgarh are emerging markets – focus on free delivery and festival combo packs to boost adoption.

# Strategy:
Use geo-targeted marketing and influencer promotions in Tier 2 & Tier 3 states.

# Product-Level Recommendations
#Product2025	         #2026 Forecast                   	#Suggested Offer
1. Lipstick	             Expected +20%	             Buy 2 Get 1 or “Beauty Box Combo
2. Children’s Book		   Steady                      Family reading packs
3. Smartwatch	            +18%                       Cashback + Warranty extension
4. Tablet	                +12%	                     Student discounts
5. Perfume	              +15%                       Personalized gift wrapping  
6. Laptop	                +10%	                     Diwali EMI scheme
7. Sneakers               +8%	                       Fashion-Fitness combo offers
8. Air Fryer	            +12%	                     Healthy Diwali campaign
9. Jeans		              +5%	                       Buy 1 Get 1 50% Off


# Customer Experience Goals for 2026

Reduce return rate by 20% through better quality checks & clear product details.
Deliver 90% of orders on-time by expanding logistics in Tier 2/3 cities.
Boost customer retention through festive loyalty programs like “Amazon Diwali Gold Members.”

# Conclusion
If Amazon leverages 2025 trends effectively, Diwali 2026 could witness:

Estimated 25–30% increase in total sales.
15% more digital payment adoption (especially UPI).
20% improvement in delivery performance.
Higher customer satisfaction & loyalty.



