from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.visualization import *
from utils.metric import *

class Exp_GBM:
    def __init__(self, args):
        self.args = args

    def simulate_gbm_standard(self, S0, mu, sigma, T, dt, num_simulations):
        """
        Simulate multiple GBM paths using standard normal distribution.
        
        Parameters:
        - S0: Initial stock price
        - mu: Expected return (drift)
        - sigma: Volatility
        - T: Time horizon (in years)
        - dt: Time step (in years, e.g., 1/252 for daily)
        - num_simulations: Number of simulation paths
        
        Returns:
        - S: Array of simulated paths (num_simulations, N)
        """
        N = int(T / dt)
        S = np.zeros((num_simulations, N))
        S[:, 0] = S0
        
        # 표준 정규분포 난수
        Z = np.random.normal(0, 1, (num_simulations, N-1))
        
        for t in range(1, N):
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
        return S

    def simulate_gbm_empirical(self, S0, mu, sigma, T, dt, num_simulations, log_returns):
        """
        Simulate multiple GBM paths using empirical distribution of log returns.
                
        """
        N = int(T / dt)
        S = np.zeros((num_simulations, N))
        S[:, 0] = S0
        
        # 경험적 분포에서 샘플링
        empirical_noise = np.random.choice(log_returns, size=(num_simulations, N-1), replace=True)
        # 노이즈를 표준화하고 sigma로 스케일링
        empirical_noise = (empirical_noise - np.mean(empirical_noise)) / np.std(empirical_noise)
        
        for t in range(1, N):
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * empirical_noise[:, t-1])
        
        return S

    def simulate_gbm(self, num_repeats=10):
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
    
        base_output_folder = os.path.join(
            './outputs/Mathematical/', f'{self.args.gbm_name}', 
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}')
                
        # 메트릭 데이터를 저장할 리스트
        metrics_list = []
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]
            print(f"\n[INFO] Simulating for ticker: {ticker_name}")
            
            stock_data = pd.read_csv(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector, stock_file))
            
            sliding_windows = get_sliding_window_data(
                stock_data,
                date_col='index',
                test_start_year=self.args.test_start_year,
                total_test_months=self.args.total_test_months,
                sliding_test_months=self.args.sliding_test_months,
                train_months=self.args.train_months
            )
            
            if len(sliding_windows) == 0:
                print(f"[WARNING] 슬라이딩 윈도우 데이터가 없습니다. {ticker_name}는 건너뜁니다.")
                continue
            
            for window_idx, (train_df, test_df, window_info) in enumerate(sliding_windows):
                print(f"[INFO] {ticker_name} - Window {window_idx+1}: {window_info['test_start']} ~ {window_info['test_end']}")
                
                # ==========================================
                # GBM parameter
                test_length = len(test_df)
                T = test_length / 252    
                dt = 1 / 252
                S0 = train_df['Adj Close'].iloc[-1]
                
                log_returns_train = np.log(train_df['Adj Close'] / train_df['Adj Close'].shift(1))[1:].values
                mu = np.mean(log_returns_train) * 252
                sigma = np.std(log_returns_train) * np.sqrt(252)
                
                # stock price time-series
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                test_close = test_df['Adj Close'].values
                
                # ==========================================
                # Monte carlo
                all_simulated_paths = []
                all_column_names = []
                
                # simulation (cross-validation)
                repeat_metrics = []
                for repeat in range(num_repeats):
                    print(f"[INFO] {ticker_name} - Window {window_idx+1} - Repeat {repeat+1}/{num_repeats}")
                    
                    if self.args.gbm_name == 'standard_gbm':
                        gbm_paths = self.simulate_gbm_standard(S0, mu, sigma, T, dt, self.args.num_simulations)
                    elif self.args.gbm_name == 'empirical_gbm':
                        gbm_paths = self.simulate_gbm_empirical(S0, mu, sigma, T, dt, self.args.num_simulations, log_returns_train)
                    
                    # Metric
                    prmsd_mean, prmsd_std = calculate_prmsd(gbm_paths, test_close)
                    coverage_ratio = calculate_coverage_ratio(gbm_paths, test_close)
                    path_diversity = calculate_path_diversity(gbm_paths)
                    ks_mean, ks_std = calculate_ks_test(gbm_paths, test_close)
                    stylized_facts = calculate_stylized_facts(gbm_paths, test_close)
                    
                    repeat_metrics.append({
                        'Repeat': repeat + 1,
                        'PRMSD_Mean': prmsd_mean,
                        'PRMSD_Std': prmsd_std,
                        'Coverage_Ratio': coverage_ratio,
                        'Path_Diversity': path_diversity,
                        'KS_Mean': ks_mean,
                        'KS_Std': ks_std,
                        'Real_Skew': stylized_facts['real_skew'],
                        'Sim_Skew_Mean': stylized_facts['sim_skew_mean'],
                        'Sim_Skew_Std': stylized_facts['sim_skew_std'],
                        'Real_Kurt': stylized_facts['real_kurt'],
                        'Sim_Kurt_Mean': stylized_facts['sim_kurt_mean'],
                        'Sim_Kurt_Std': stylized_facts['sim_kurt_std'],
                        'Real_Hill': stylized_facts['real_hill'],
                        'Sim_Hill_Mean': stylized_facts['sim_hill_mean'],
                        'Sim_Hill_Std': stylized_facts['sim_hill_std']
                    })
                    
                    all_simulated_paths.append(gbm_paths.T)  # Transpose to have time steps as rows
                    all_column_names.extend([f"repeat_{repeat+1}_S_{i+1}" for i in range(self.args.num_simulations)])
                    
                    # Save results (visualization & csv)
                    simulation_folder = os.path.join(base_output_folder, 'GBM_scenario_4', self.args.exp_root_path, self.args.sector, ticker_name)
                    os.makedirs(simulation_folder, exist_ok=True)
                    
                    plot_simulations(
                        model_name=self.args.gbm_name,
                        real_prices=real_prices,
                        fake_prices=None,
                        test_close=test_close,
                        simulated_S=gbm_paths,
                        num_plot=self.args.num_simulations,
                        save_path=os.path.join(simulation_folder, f"Sliding_Window_{window_idx+1}_Repeat_{repeat+1}_simulation_result.png")
                    )
                    print(f"[INFO] Window {window_idx+1} - Repeat {repeat+1}: 시각화 저장 완료")
                
                # concat the results of cross-validation
                all_simulated_df = pd.DataFrame(
                    np.concatenate(all_simulated_paths, axis=1),
                    columns=all_column_names
                )
                
                # single csv
                simulation_file = os.path.join(simulation_folder, f"Sliding_Window_{window_idx+1}_simulated_paths.csv")
                all_simulated_df.to_csv(simulation_file, index=False)
                print(f"[INFO] Window {window_idx+1}: 모든 시뮬레이션 경로를 {simulation_file}에 저장 완료")
                
                # 반복 메트릭 집계 (평균 및 표준편차)
                repeat_metrics_df = pd.DataFrame(repeat_metrics)
                aggregated_metrics = {
                    'Ticker': ticker_name,
                    'Window': window_idx + 1,
                    'Test_Start': window_info['test_start'],
                    'Test_End': window_info['test_end'],
                    'PRMSD_Mean_Avg': repeat_metrics_df['PRMSD_Mean'].mean(),
                    'PRMSD_Mean_Std': repeat_metrics_df['PRMSD_Mean'].std(),
                    'PRMSD_Std_Avg': repeat_metrics_df['PRMSD_Std'].mean(),
                    'Coverage_Ratio_Avg': repeat_metrics_df['Coverage_Ratio'].mean(),
                    'Coverage_Ratio_Std': repeat_metrics_df['Coverage_Ratio'].std(),
                    'Path_Diversity_Avg': repeat_metrics_df['Path_Diversity'].mean(),
                    'Path_Diversity_Std': repeat_metrics_df['Path_Diversity'].std(),
                    'KS_Mean_Avg': repeat_metrics_df['KS_Mean'].mean(),
                    'KS_Mean_Std': repeat_metrics_df['KS_Mean'].std(),
                    'KS_Std_Avg': repeat_metrics_df['KS_Std'].mean(),
                    'Real_Skew': repeat_metrics_df['Real_Skew'].mean(),  # 실제 데이터는 동일
                    'Sim_Skew_Mean_Avg': repeat_metrics_df['Sim_Skew_Mean'].mean(),
                    'Sim_Skew_Mean_Std': repeat_metrics_df['Sim_Skew_Mean'].std(),
                    'Sim_Skew_Std_Avg': repeat_metrics_df['Sim_Skew_Std'].mean(),
                    'Real_Kurt': repeat_metrics_df['Real_Kurt'].mean(),
                    'Sim_Kurt_Mean_Avg': repeat_metrics_df['Sim_Kurt_Mean'].mean(),
                    'Sim_Kurt_Mean_Std': repeat_metrics_df['Sim_Kurt_Mean'].std(),
                    'Sim_Kurt_Std_Avg': repeat_metrics_df['Sim_Kurt_Std'].mean(),
                    'Real_Hill': repeat_metrics_df['Real_Hill'].mean(),
                    'Sim_Hill_Mean_Avg': repeat_metrics_df['Sim_Hill_Mean'].mean(),
                    'Sim_Hill_Mean_Std': repeat_metrics_df['Sim_Hill_Mean'].std(),
                    'Sim_Hill_Std_Avg': repeat_metrics_df['Sim_Hill_Std'].mean()
                }
                
                # 메트릭 리스트에 추가
                metrics_list.append(aggregated_metrics)
            
        # 메트릭 저장
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_folder = os.path.join(base_output_folder, 'GBM_metric_cross_validated_2', self.args.exp_root_path)
            os.makedirs(metrics_folder, exist_ok=True)
            metrics_df.to_csv(os.path.join(metrics_folder, f"{self.args.sector}_GBM_metric_cross_validated.csv"), index=False)
            print(f"[INFO] 모든 평가 지표 저장 완료: {os.path.join(metrics_folder, f'{self.args.sector}_GBM_metric_cross_validated.csv')}")
        else:
            print("[WARNING] 저장할 메트릭 데이터가 없습니다.")