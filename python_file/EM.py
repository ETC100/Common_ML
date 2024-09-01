import numpy as np

# True probability
p_A_true = 0.7
p_B_true = 0.5

# generate data
def generate_data(p_A, p_B, num_samples):
    data = []
    for _ in range(num_samples):
        # 随机选择硬币
        if np.random.rand() > 0.5:  # choose A/B = 1:1
            coin = 'A'
            heads = np.random.binomial(n=10, p=p_A)  # toss 10 times
        else:  # 有50%的概率选择硬币B
            coin = 'B'
            heads = np.random.binomial(n=10, p=p_B)  # toss 10 times
        tails = 10 - heads
        data.append([heads, tails])
    return np.array(data)

np.random.seed(42)
observations = generate_data(p_A_true, p_B_true, num_samples=100)

# initialization
p_A = 0.6
p_B = 0.5

# EM algorithm
def em_algorithm(observations, p_A, p_B, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        # E step
        expectations = []
        for obs in observations:
            n_heads, n_tails = obs
            weight_A = (p_A ** n_heads) * ((1 - p_A) ** n_tails)
            weight_B = (p_B ** n_heads) * ((1 - p_B) ** n_tails)
            weight_sum = weight_A + weight_B
            expectations.append([weight_A / weight_sum, weight_B / weight_sum])
        expectations = np.array(expectations)
        

        print(f"Iteration {i+1}: Expectations (weight_A, weight_B) = {expectations[:5]}")

        # M step
        new_p_A = np.sum(expectations[:, 0] * observations[:, 0]) / np.sum(expectations[:, 0] * np.sum(observations, axis=1))
        new_p_B = np.sum(expectations[:, 1] * observations[:, 0]) / np.sum(expectations[:, 1] * np.sum(observations, axis=1))
        
        print(f"Iteration {i+1}: p_A = {new_p_A}, p_B = {new_p_B}")

        if abs(new_p_A - p_A) < tol and abs(new_p_B - p_B) < tol:
            break

        p_A, p_B = new_p_A, new_p_B

    return p_A, p_B

p_A, p_B = em_algorithm(observations, p_A, p_B)
print(f"Estimated p_A: {p_A}")
print(f"Estimated p_B: {p_B}")
