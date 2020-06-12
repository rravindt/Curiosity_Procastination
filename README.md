# Curiosity_Procastination
We have implemented the curiosity based RL model which deals with the noisy TV problem faced by it.

The following program can be run after installing the following libries:

keras
tensorflow
matplotlib
pickle
gym

After the installation of the following libraries then the file can run.

There are two files compiled - "Curiosity.py" which is the orginal compilation of the curiosity algorithm
							 - "Curiosity_Mod.py" which is our implementation if the curiosity model.

The two files can be run as follows:

		python Curiosity_Mod.py --alpha=<alpha_value> --noisy=<noisy_value>

		python Curiosity.py --noisy=<noisy_value>

We can specify the value of alpha in <alpha_value> as an argument.
We can specify the value of the noisy TV i.e, <noisy_value> = 0, means that the noisy tv is not active
											  <noisy_value> = 1, means that the noisy tv method 1 is active
											  <noisy_value> = 2, means that the noisy tv method 2 is active
