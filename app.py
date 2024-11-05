from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key for session management

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color="blue", label="Data Points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)

        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(s >= slope for s in slopes) / S
    intercept_extreme = sum(i >= intercept for i in intercepts) / S

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Initialize p_value
    p_value = None

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "≠":
        p_value = 2 * min(np.mean(simulated_stats >= observed_stat), np.mean(simulated_stats <= observed_stat))

    # Set a fun message if p_value is very small
    fun_message = "Rare event detected!" if p_value is not None and p_value <= 0.0001 else None

    # Plot histogram of simulated statistics
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, alpha=0.5, color="gray", label="Simulated Statistics")
    plt.axvline(observed_stat, color="blue", linestyle="--", label=f"Observed {parameter}")
    plt.axvline(hypothesized_value, color="red", linestyle="--", label="Hypothesized Value")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s with Observed and Hypothesized Values")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value if p_value is not None else "N/A",  # Add default value
        fun_message=fun_message
    )



@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    S = int(session.get("S"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    estimates = np.array(slopes if parameter == "slope" else intercepts)
    true_param = beta1 if parameter == "slope" else beta0

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # Calculate confidence interval for the parameter estimate
    t_value = t.ppf((1 + confidence_level) / 2, df=S - 1)
    ci_lower = mean_estimate - t_value * std_estimate
    ci_upper = mean_estimate + t_value * std_estimate

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot individual estimates and confidence interval
    plt.figure(figsize=(8, 6))
    plt.scatter(range(S), estimates, color="gray", alpha=0.5, label="Simulated Estimates")
    plt.axhline(mean_estimate, color="blue", linestyle="--", label="Mean Estimate")
    plt.axhline(ci_lower, color="red", linestyle="--", label="CI Lower Bound")
    plt.axhline(ci_upper, color="green", linestyle="--", label="CI Upper Bound")
    plt.title(f"Confidence Interval for {parameter.capitalize()}")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true
    )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
