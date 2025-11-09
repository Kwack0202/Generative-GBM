from common_imports import *

def calculate_prmsd(simulated_paths, real_prices):
    """
    Calculate Percent Root Mean Square Difference (PRMSD) for each path.
    Returns mean and standard deviation of PRMSD across all paths, rounded to 3 decimal places.
    """
    prmsd_values = []
    mean_real = np.mean(real_prices)
    
    for i in range(simulated_paths.shape[0]):
        mse = np.mean((simulated_paths[i, :] - real_prices) ** 2)
        rmse = np.sqrt(mse)
        prmsd = (rmse / mean_real) * 100
        prmsd_values.append(prmsd)
    
    return np.round(np.mean(prmsd_values), 3), np.round(np.std(prmsd_values), 3)

def calculate_coverage_ratio(simulated_paths, real_prices, confidence_level=0.95):
    """
    Calculate Coverage Ratio by checking how often real prices fall within the 95% confidence interval
    of simulated paths at each time step, rounded to 3 decimal places.
    """
    # 각 시간 스텝에서 시뮬레이션 경로의 95% 신뢰구간 계산
    lower_bound = np.percentile(simulated_paths, (1 - confidence_level) / 2 * 100, axis=0)
    upper_bound = np.percentile(simulated_paths, (1 + confidence_level) / 2 * 100, axis=0)
    
    # 실제 주가가 신뢰구간 내에 포함되는지 확인
    coverage = np.sum((real_prices >= lower_bound) & (real_prices <= upper_bound))
    
    # 커버리지 비율 계산 (포인트 수로 나눔)
    coverage_ratio = coverage / len(real_prices)
    
    return np.round(coverage_ratio, 3)

def calculate_path_diversity(simulated_paths):
    """
    Calculate Path Diversity using Mean Squared Distance (MSD) between paths, rounded to 3 decimal places.
    """
    num_paths = simulated_paths.shape[0]
    msd_sum = 0
    count = 0
    
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            msd_sum += np.mean((simulated_paths[i, :] - simulated_paths[j, :]) ** 2)
            count += 1
    
    msd = np.sqrt(msd_sum / count) if count > 0 else 0
    return np.round(msd, 3)

def calculate_ks_test(simulated_paths, real_prices):
    """
    Calculate Kolmogorov-Smirnov test statistic for log returns.
    Returns mean and std of KS statistics across all paths, rounded to 3 decimal places.
    """
    real_log_returns = np.log(real_prices[1:] / real_prices[:-1])
    ks_stats = []
    
    for i in range(simulated_paths.shape[0]):
        sim_prices = simulated_paths[i, :]
        sim_log_returns = np.log(sim_prices[1:] / sim_prices[:-1])
        ks_stat, _ = stats.ks_2samp(real_log_returns, sim_log_returns)
        ks_stats.append(ks_stat)
    
    return np.round(np.mean(ks_stats), 3), np.round(np.std(ks_stats), 3)

def calculate_stylized_facts(simulated_paths, real_prices):
    """
    Calculate Stylized Facts (skewness, kurtosis, Hill estimator for tail index).
    Returns mean and std of each metric across all paths, plus real data values, rounded to 3 decimal places.
    """
    real_log_returns = np.log(real_prices[1:] / real_prices[:-1])
    real_skew = np.round(stats.skew(real_log_returns), 3)
    real_kurt = np.round(stats.kurtosis(real_log_returns, fisher=True), 3)
    
    # Hill estimator for tail index (극단값 분석)
    sorted_returns = np.sort(np.abs(real_log_returns))[::-1]
    k = int(len(sorted_returns) * 0.05)  # 상위 5% 사용
    real_hill = np.round(1 / np.mean(np.log(sorted_returns[:k] / sorted_returns[k])), 3) if k > 0 else 0
    
    skews, kurts, hills = [], [], []
    
    for i in range(simulated_paths.shape[0]):
        sim_prices = simulated_paths[i, :]
        sim_log_returns = np.log(sim_prices[1:] / sim_prices[:-1])
        
        skews.append(stats.skew(sim_log_returns))
        kurts.append(stats.kurtosis(sim_log_returns, fisher=True))
        
        sorted_sim = np.sort(np.abs(sim_log_returns))[::-1]
        k = int(len(sorted_sim) * 0.05)
        hills.append(1 / np.mean(np.log(sorted_sim[:k] / sorted_sim[k])) if k > 0 else 0)
    
    return {
        'real_skew': real_skew,
        'sim_skew_mean': np.round(np.mean(skews), 3),
        'sim_skew_std': np.round(np.std(skews), 3),
        'real_kurt': real_kurt,
        'sim_kurt_mean': np.round(np.mean(kurts), 3),
        'sim_kurt_std': np.round(np.std(kurts), 3),
        'real_hill': real_hill,
        'sim_hill_mean': np.round(np.mean(hills), 3),
        'sim_hill_std': np.round(np.std(hills), 3)
    }