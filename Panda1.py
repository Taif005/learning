import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv('home_prices.csv')

plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.scatter(df["Square_Feet"],df["Price"])


reg = linear_model.LinearRegression()
reg.fit(df[["Square_Feet"]],df.Price)

# Plot actual data points
plt.scatter(
    df["Square_Feet"],
    df["Price"],
    color="royalblue",
    alpha=0.6,
    label="Actual Data",
)

# Overlay the regression line
plt.plot(
    df["Square_Feet"],
    reg.predict(df[["Square_Feet"]]),
    color="red",
    linewidth=2,
    label="Regression Line",
)

plt.xlabel("Square Feet")
plt.ylabel("Price ($)")
plt.title("Home Price vs. Square Feet")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()
