import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# global plot styling: cleaner font and roomier spacing for all figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.titlepad': 12,
    'axes.labelsize': 12,
    'axes.labelpad': 8,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# import the data from a CSV file
data = pd.read_csv('oliviastats.csv')
daily_cstats = data['Daily Cost']
days = data['Day']
# initial data exploration
print(data.head())
print(data.describe())
cost_mean = data.describe().loc['mean', 'Daily Cost']
cost_std = data.describe().loc['std', 'Daily Cost']
print(f"Mean Daily Cost: ${cost_mean:.2f}")
print(f"Standard Deviation of Daily Cost: ${cost_std:.2f}")

# hypothesized mean for the null hypothesis, defined up front so it can be
# drawn on the histogram below as well as used in the hypothesis test later
hypothesized_mean = 14.29  # example hypothesized mean

class OneVarStats:
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data, ddof=1)  # sample standard deviation
        self.n = len(data)

    def confidence_interval(self, confidence_level=0.95):
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * (self.std / np.sqrt(self.n))
        return (self.mean - margin_of_error, self.mean + margin_of_error)

    def hypothesis_test(self, hypothesized_mean, alpha=0.05):
        z_score = (self.mean - hypothesized_mean) / (self.std / np.sqrt(self.n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        reject_null = p_value < alpha
        # hypothesisresult holds the message string and is printed here, so it
        # prints to the console automatically every time this function is called
        if reject_null:
            hypothesisresult = f"Reject the null hypothesis (p-value = {p_value:.4f})"
        else:
            hypothesisresult = f"Accept the null hypothesis (p-value = {p_value:.4f})"
        print(hypothesisresult)
        return reject_null, p_value
        # returns True if we reject the null hypothesis, plus the p-value


class TwoVarStats:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
        self.mean_x = np.mean(xdata)
        self.mean_y = np.mean(ydata)
        self.std_x = np.std(xdata, ddof=1)
        self.std_y = np.std(ydata, ddof=1)
        self.variance_x = np.var(xdata, ddof=1)
        self.variance_y = np.var(ydata, ddof=1)
        self.covariance_xy = np.cov(xdata, ydata)[0][1]
        self.slope = self.covariance_xy / self.variance_x
        self.n = len(xdata)
        self.r_squared = (self.covariance_xy ** 2) / (self.variance_x * self.variance_y)
    def confidence_interval_slope(self, confidence_level=0.95):
        # use ppf (percent-point/quantile function), not pdf (density function):
        # we need the t critical value at the given confidence level (the x
        # such that P(T <= x) = confidence), which is what ppf returns.
        # pdf would instead give the density (height of the curve) at that
        # point, which is the wrong quantity for a margin of error.
        t_score = stats.t.ppf((1 + confidence_level) / 2, df=self.n - 2)
        margin_of_error = t_score * (self.std_y / np.sqrt(self.n * self.variance_x))
        return (self.slope - margin_of_error, self.slope + margin_of_error)
    def graph(self, ax=None):
        # ax lets this be drawn onto an existing subplot (e.g. side-by-side
        # with the histogram); if no ax is passed, it makes its own figure
        # and shows it immediately, same as before
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.xdata, self.ydata, color='steelblue', edgecolor='white', s=50, alpha=0.85, zorder=3)
        ax.plot(self.xdata, self.slope * self.xdata + self.mean_y - self.slope * self.mean_x,
                color='crimson', linewidth=2, zorder=2)
        ax.set_xlabel('Day')
        ax.set_ylabel('Daily Cost')
        ax.set_title('Daily Cost vs Day (R² = {:.2f})'.format(self.r_squared))
        ax.grid(True, alpha=0.25, linestyle='--')
        if standalone:
            plt.show()

# try it out
oliviastats = OneVarStats(daily_cstats)
print(f"95% Confidence Interval for the Mean Daily Cost: ${oliviastats.confidence_interval()[0]:.2f} to ${oliviastats.confidence_interval()[1]:.2f}")
print(f"Mean: ${oliviastats.mean:.2f}, Standard Deviation: ${oliviastats.std:.2f}, Sample Size: {oliviastats.n}")
# try it out

# try it out
twovarstats = TwoVarStats(daily_cstats, days)
print(twovarstats.confidence_interval_slope())

# histogram of daily costs and the Day-vs-Cost scatter/regression, side by
# side in one figure
fig, (hist_ax, scatter_ax) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig.suptitle('Olivia\'s Daily Cost Analysis', fontsize=16, fontweight='bold')

hist_ax.hist(daily_cstats, bins=30, edgecolor='white', color='steelblue', alpha=0.85, label='Daily Cost')
# vertical bar in a different color marking the null hypothesis mean, so you
# can see at a glance where it falls relative to the observed distribution
hist_ax.axvline(hypothesized_mean, color='crimson', linewidth=3,
                 label=f'Null Hypothesis Mean (${hypothesized_mean:.2f})')
hist_ax.set_xlabel('Daily Cost')
hist_ax.set_ylabel('Frequency')
hist_ax.set_title('Distribution of Daily Costs')
hist_ax.grid(True, axis='y', alpha=0.25, linestyle='--')
hist_ax.legend()

twovarstats.graph(ax=scatter_ax)

plt.show()

# try out hypothesis testing
# perform hypothesis test using daily_cstats, having mean of oliviastats.mean and the hypothesized mean
reject_null, p_value = oliviastats.hypothesis_test(hypothesized_mean)
