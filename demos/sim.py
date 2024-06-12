
# is defined as :math:`f(x) = x \sin(x)`.
import numpy as np



import matplotlib.pyplot as plt

X_train = np.array([0.258790016, -0.562828779, 2.040450096, 0.536149979, -0.273379803, 0.140659809, -0.362749815, 0.940116167, -1.13737011, 2.511609554, 0.648459911, 1.442019939, 1.242062092, 2.743859768, 1.203340054, 0.599410057, 2.189549923, -0.153599739, 0.28662014, -2.911369801, 0.582506895, 0.001574993, 2.80185008, 0.824699879, 0.965509653, 1.861440182, -3.569489956, 0.045530319, -0.43143034, 0.969830036, 1.311029911, -4.714150429, 2.520063519, 2.099699855, -0.67315197, 1.984168768, 0.246829987, 0.656771183, -1.158186436, 1.434120178, 2.248100281, 1.523965836, 1.528709888, 0.708849907, 1.087829828, 1.266985178, -0.047230005, 3.419819832, -1.172410011, 0.338779926, 1.282601357, -0.312160015, -2.256960154, -0.130310059, -0.21295023, -0.028310061, 0.35255003, 3.080460072, -1.30776, -1.485800266, 0.805532932, 2.585520267, 0.000442266, -0.013020039, 1.129889965, -0.085253239, -0.661720276, 0.048409939, 0.850209951, -1.534351349, -1.582495213, 0.320583344, 0.607164383, 1.181360006, 2.343709946, 1.161499977, -1.495592833, -0.590740204, -0.207507133, 0.080735683, 3.393496037, 0.517369986, -0.076980114, 1.857439995, 1.214299917, 2.324719906, -0.373440504, 2.594691277, 0.695320129, 0.716680527, -0.676769733, -0.404969692, -0.71724987, 0.761677504, 0.000479937, 0.675540447, -4.615100384, -0.60199976, 0.422535181, 3.690630198, 0.171619892, -0.257740021, 0.100880146, 1.266279936, 0.653150082, 1.144410133, 0.666920185, 2.781790257, -0.333880186, 0.103044987, 1.563940048, 0.680729866, 1.029200077, 0.510810733, 0.142250061, 0.465650082, -0.650209427, 1.556699991, -1.518379927, -1.670211315, -1.20871973, 1.883759975, 0.740951538]).reshape(-1,1)
y_train = np.array([-0.5, 9.7, 0.7, -11.4, -0.5, 0.7, -12.4, -27.7, -19.2, -0.4, -33, -0.9, -16.6, -26.4, -12.8, -5.6, -10, -6, 0, -2, 3.8, -5.7, -14.7, -12, -14.2, -12.7, -15.6, -0.6, -0.1, 0, -0.3, -3.4, -5.6, -18.6, -2.1, -17.6, -2.2, -3.8, -0.4, 0, 0.1, -33.5, -11.5, 0, -33.2, 0.3, -2.2, -11.6, 0.3, 0, -28, -8.2, -0.6, 0.3, 0.6, -3.4, 0.2, 0.4, -7.9, 9.7, -0.3, -36.9, -16.6, -2, -21.3, 1.7, -2.5, 0, -2.7, -3.6, -6.6, -0.2, -11.6, -0.2, -6, -33.5, -27.7, -4.8, 9.8, -4.2, -20.3, -32.1, -0.5, -21.9, -33.5, 0, -5.1, 0.1, -16.5, -11.5, -2.1, -0.4, -6, -3, -23.2, -1, -7, -3.7, -0.7, -35.2, 0, 0, -13, 0.4, -15.6, 0.2, -0.7, -1.4, -11.3, -22.7, -32.6, -12.2, -2.6, 4.7, -2.7, 0.5, -0.5, -0.5, 0.4, -10.4, 0.5, 3.4, 12.6]).reshape(-1,1)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

kernel = RationalQuadratic(length_scale= 100, alpha=1.5, length_scale_bounds=(1e-10,1e1))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
X_test = np.linspace(start=-4, stop=4, num=1_000).reshape(-1, 1)
mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)

#plt.scatter(X_train, y_train, label="Observations")
plt.plot(X_test, mean_prediction, label="Mean prediction")
plt.fill_between(
    X_test.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
    color='#F97305'
)
plt.legend()
plt.xlabel("Change in Education(% of GDP)")
plt.ylabel("Change in Poverty Rate (% of population)")
plt.title("Education Funding vs Change in Poverty")
plt.ylim(-20,0)
plt.show()
"""

# %%
# We see that for a prediction made on a data point close to the one from the
# training set, the 95% confidence has a small amplitude. Whenever a sample
# falls far from training data, our model's prediction is less accurate and the
# model prediction is less precise (higher uncertainty).
#
# Example with noisy targets
# --------------------------
#
# We can repeat a similar experiment adding an additional noise to the target
# this time. It will allow seeing the effect of the noise on the fitted model.
#
# We add some random Gaussian noise to the target with an arbitrary
# standard deviation.
noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

# %%
# We create a similar Gaussian process model. In addition to the kernel, this
# time, we specify the parameter `alpha` which can be interpreted as the
# variance of a Gaussian noise.
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# %%
# Let's plot the mean prediction and the uncertainty region as before.
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on a noisy dataset")

# %%
# The noise affects the predictions close to the training samples: the
# predictive uncertainty near to the training samples is larger because we
# explicitly model a given level target noise independent of the input
# variable.
"""