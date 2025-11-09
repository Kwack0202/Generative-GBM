from common_imports import *
from GBM.exp.exp_basic import Exp_Basic
from GBM.exp.data_loader import StockDataset

from GBM.models.VanillaGAN import Generator, Discriminator
from GBM.models.WGAN import WGenerator, WDiscriminator
from GBM.models.QuantGAN import QuantGenerator, QuantDiscriminator

from utils.sliding_window import get_sliding_window_data
from utils.gmm_preprecessing import *
from utils.visualization import *
from utils.metric import *
import copy

np.seterr(all='warn')  # 모든 연산에 대해 경고 발생하도록 설정
warnings.filterwarnings('ignore', category=RuntimeWarning) 

class Exp_GAN(Exp_Basic):
    def __init__(self, args):
        super(Exp_GAN, self).__init__(args)

        self.netG = None  
        self.log_returns_mean = None
        self.log_returns_max = None
        self.params = None 
    
    def _get_data(self, df):
        log_returns_mean = np.mean(df)
        log_return_norm = df - log_returns_mean
        params = igmm(log_return_norm)
        log_processed = W_delta((log_return_norm - params[0]) / params[1], params[2])
        log_returns_max = np.max(np.abs(log_processed))
        log_processed /= log_returns_max
        
        return log_returns_mean, params, log_returns_max, log_processed

    def _build_generator(self):
        if self.args.model_name == 'VanillaGAN':
            netG = Generator(self.args.noise_input_size, self.args.noise_output_size).to(self.device)
        elif self.args.model_name == 'WGAN':
            netG = WGenerator(self.args.noise_input_size, self.args.noise_output_size).to(self.device)
        elif self.args.model_name == 'QuantGAN':
            netG = QuantGenerator(self.args.noise_input_size, self.args.noise_output_size).to(self.device)
        return netG

    def _build_discriminator(self):
        if self.args.model_name == 'VanillaGAN':
            netD = Discriminator(self.args.noise_output_size).to(self.device)
        elif self.args.model_name == 'WGAN':
            netD = WDiscriminator(self.args.noise_output_size).to(self.device)
        elif self.args.model_name == 'QuantGAN':
            netD = QuantDiscriminator(self.args.noise_output_size, self.args.noise_output_size).to(self.device)
        return netD
            
    def _select_optimizer(self, netG, netD):
        if self.args.model_optimizer == 'Adam':
            optG = optim.Adam(netG.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
            optD = optim.Adam(netD.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
            
        elif self.args.model_optimizer == 'RMSprop':
            optG = optim.RMSprop(netG.parameters(), lr=self.args.learning_rate)
            optD = optim.RMSprop(netD.parameters(), lr=self.args.learning_rate)
            
        else:
            raise ValueError("Unknown optimizer specified")
        
        return optG, optD
    
    def _save_settings(self):
        """
        Save the values of all parser arguments passed in run.py to the settings.txt file.
        """
        settings_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
        
        os.makedirs(settings_folder, exist_ok=True)
        settings_path = os.path.join(settings_folder, f'settings.txt')
        with open(settings_path, 'w') as f:
            for key, value in vars(self.args).items():
                f.write(f"{key}: {value}\n")
        
    def _calculate_mmd(self, real_data, fake_data, sigma=1.0):
        """
        Calculate MMD between real and fake data using Gaussian kernel.
        Returns MMD score rounded to 3 decimal places.
        """
        real_data = np.asarray(real_data).reshape(-1, 1)
        fake_data = np.asarray(fake_data).reshape(-1, 1)
        
        n, m = len(real_data), len(fake_data)
        xx = np.exp(-cdist(real_data, real_data, metric='sqeuclidean') / (2 * sigma**2))
        yy = np.exp(-cdist(fake_data, fake_data, metric='sqeuclidean') / (2 * sigma**2))
        xy = np.exp(-cdist(real_data, fake_data, metric='sqeuclidean') / (2 * sigma**2))
        
        mmd = np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
        mmd = np.sqrt(max(mmd, 0))
        return np.round(mmd, 3)
    
    def train_gan(self):
        self._save_settings()
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
        
        base_output_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
        
        metrics_list = []
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]
            print(f'\nGenerativeAI-GBM start ~~ Stock code: {ticker_name}')
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
                print(f"\n--- Sliding Window {window_idx+1} ---")
                print(f"  Train period: {window_info['train_start']} ~ {window_info['train_end']} | Rows: {window_info['train_rows']}")
                print(f"  Test period: {window_info['test_start']} ~ {window_info['test_end']} | Rows: {window_info['test_rows']}")

                # step 1. Data preprocessing (GMM)
                log_returns_train = np.log(train_df['Adj Close'] / train_df['Adj Close'].shift(1))[1:].values
                log_returns_mean, params, log_returns_max, log_processed = self._get_data(log_returns_train)
                
                self.log_returns_mean = log_returns_mean
                self.params = params
                self.log_returns_max = log_returns_max
                
                # step 2. Model define
                netG = self._build_generator()
                netD = self._build_discriminator()
                optG, optD = self._select_optimizer(netG, netD)
                
                # step 3. Dataset load
                dataset = StockDataset(log_processed, self.args.seq_len)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.args.num_workers)
                
                # step 4. Model train   
                progress = tqdm(range(self.args.num_epochs))
                
                best_loss = float('inf')
                early_stop_counter = 0
                best_epoch = 0
                
                best_netG = None
                best_netD = None
                best_ks_stat = None
                best_mmd_score = None
                
                prev_loss = float('inf')
                
                for epoch in progress:
                    for i, data in enumerate(dataloader, 0):
                        netD.zero_grad()
                        real = data.to(self.device)
                        batch_size, seq_len = real.size(0), real.size(1)
                        noise = torch.randn(batch_size, seq_len, self.args.noise_input_size, device=self.device)
                        fake = netG(noise).detach()

                        if self.args.model_name == 'VanillaGAN':
                            lossD = -(torch.mean(torch.log(netD(real))) + torch.mean(torch.log(1. - netD(fake))))
                            lossD.backward()
                            optD.step()
                        elif self.args.model_name in ['WGAN', 'QuantGAN']:
                            lossD = -torch.mean(netD(real)) + torch.mean(netD(fake))
                            lossD.backward()
                            optD.step()
                            for p in netD.parameters():
                                p.data.clamp_(-0.01, 0.01)
                                            
                        if i % 5 == 0:
                            netG.zero_grad()
                            if self.args.model_name == 'VanillaGAN':
                                lossG = torch.mean(torch.log(1. - netD(netG(noise))))
                            elif self.args.model_name in ['WGAN', 'QuantGAN']:
                                lossG = -torch.mean(netD(netG(noise)))
                            lossG.backward()
                            optG.step()
                    
                    # Generate fake data for coverage ratio calculation
                    generated_noise = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(log_returns_train), nz=self.args.noise_input_size, generator=netG)
                    real_noise = log_processed.flatten()
                    
                    # Calculate KS statistic and MMD score
                    ks_stat, ks_pvalue = ks_2samp(real_noise, generated_noise)
                    mmd_score = self._calculate_mmd(real_noise, generated_noise, sigma=1.0)
                        
                    # progress description    
                    msg = (f'Loss_D: {lossD.item():.8f} | Loss_G: {lossG.item():.8f} | KS statistic: {ks_stat:.6f} (p-value: {ks_pvalue:.4f}) | MMD Score: {mmd_score:.3f} ')
                    progress.set_description(msg)
                    
                    # Monitoring & EarlyStop
                    if (epoch + 1) >= self.args.min_epochs:
                        if (lossG.item() < prev_loss - self.args.loss_tolerance and
                            ks_stat < 0.05 and
                            ks_pvalue > 0.05 and
                            mmd_score < 0.05):  
                            best_loss = lossG.item()
                            best_epoch = epoch + 1
                            best_netG = copy.deepcopy(netG)
                            best_netD = copy.deepcopy(netD)
                            best_ks_stat = ks_stat
                            best_mmd_score = mmd_score
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if (early_stop_counter >= self.args.early_stop_patience and
                                ks_stat < 0.05 and
                                ks_pvalue > 0.05 and
                                mmd_score < 0.05):
                                print(f"\n[Early Stopping] Epoch {epoch+1}: No improvement in coverage ratio for {early_stop_counter} consecutive epochs.")
                                break
                        
                    prev_loss = lossG.item()
                
                if best_netG is not None:
                    print(f"\n[Best INFO] Epoch: {best_epoch} | Generator Loss: {best_loss:.8f} | KS Statistic: {best_ks_stat:.6f} | MMD Score: {best_mmd_score:.3f}")
                    netG = best_netG
                    netD = best_netD
                    final_epoch = best_epoch
                else:
                    final_epoch = epoch + 1
                
                output_folder = os.path.join(base_output_folder, f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/')
                os.makedirs(output_folder, exist_ok=True)
                
                netG_save_path = os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_netG_epoch_{final_epoch}.pth")
                netD_save_path = os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_netD_epoch_{final_epoch}.pth")
                torch.save(netG, netG_save_path)
                torch.save(netD, netD_save_path)
                
                # step 5. Simulation
                self.netG = torch.load(netG_save_path)
                self.netG.eval()

                '''
                Generate Fake Data using GAN (Train period)
                '''
                real_cumsum = log_returns_train.cumsum()
                fakes_cumsum = self.generate_fakes(log_returns_train, self.args.noise_input_size, cumsum=True).flatten()
                
                P0 = train_df['Adj Close'].iloc[0]
                real_prices = P0 * np.exp(real_cumsum)
                fake_prices = P0 * np.exp(fakes_cumsum)
                
                '''
                Generate Confidence coverage using GAN (Train period)
                '''
                fake_data = self.generate_fakes(log_returns_train, self.args.noise_input_size, cumsum=True, n=self.args.fake_sample)
                fake_array = P0 * np.exp(fake_data.values)
                
                conf_levels = [0.5, 0.95, 0.99]
                bounds = {}
                for conf in conf_levels:
                    alpha = (1 - conf) / 2 * 100
                    bounds[conf] = {
                        'lower': np.percentile(fake_array, alpha, axis=1),
                        'upper': np.percentile(fake_array, 100 - alpha, axis=1)
                    }
                
                '''
                Generate Fake Data using GBM (Test period)
                '''         
                test_length = len(test_df)
                T = test_length / 252
                dt = 1 / 252

                test_close = test_df['Adj Close'].values
                S0_price = train_df['Adj Close'].iloc[-1]
                
                gbm_paths = self.simulate_gbm_GAN(
                    S0=S0_price,
                    mu=np.mean(log_returns_train) * 252,
                    sigma=np.std(log_returns_train) * np.sqrt(252),
                    T=T,
                    dt=dt,
                    generator=self.netG,
                    nz=self.args.noise_input_size,
                    num_simulations=self.args.num_simulations,
                )

                '''
                Plot Simulations and Confidence Intervals
                ''' 
                plot_simulations(
                    real_prices=real_prices,
                    fake_prices=fake_prices,
                    test_close=test_close,
                    simulated_S=gbm_paths,
                    num_plot=self.args.num_simulations,
                    save_path=os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_simulation_result.png")
                )
                
                plot_confidence_interval(
                    real_prices=real_prices,
                    bounds=bounds,
                    test_close=test_close,
                    simulated_S=gbm_paths,
                    num_plot=self.args.num_simulations,
                    save_path=os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_confidence_results.png")
                )

                learned_noise = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(log_returns_train), nz=self.args.noise_input_size, device=self.device)
                real_noise = log_processed.flatten()
                
                final_ks_stat, final_ks_pvalue = ks_2samp(real_noise, learned_noise)
                final_mmd_score = self._calculate_mmd(real_noise, learned_noise, sigma=1.0)
                
                # Calculate coverage ratio for 95% confidence interval
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                
                fake_data = self.generate_fakes(log_returns_train, self.args.noise_input_size, cumsum=True, n=self.args.fake_sample, generator=netG)
                fake_array = P0 * np.exp(fake_data.values)
                
                conf_level = 0.95
                alpha = (1 - conf_level) / 2 * 100
                lower_bound = np.percentile(fake_array, alpha, axis=1)
                upper_bound = np.percentile(fake_array, 100 - alpha, axis=1)
                
                # Calculate coverage ratio
                within_bounds = (real_prices >= lower_bound) & (real_prices <= upper_bound)
                final_coverage_ratio = np.mean(within_bounds)
                
                # Calculate path diversity
                final_path_diversity = calculate_path_diversity(fake_array)
                    
                visualize_noise_distribution(
                    real_noise, learned_noise,
                    hist_save_path=os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_noise_distribution.png"),
                    qq_save_path=os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_qq_plot.png")
                )
                
                metrics = {
                    'Ticker': ticker_name,
                    'Window': window_idx + 1,
                    'Train_Start': window_info['train_start'],
                    'Train_End': window_info['train_end'],
                    'Final_Epoch': final_epoch,
                    'KS_Stat': np.round(final_ks_stat, 3),
                    'KS_Pvalue': np.round(final_ks_pvalue, 3),
                    'MMD_Score': final_mmd_score,
                    'Coverage_Ratio' : final_coverage_ratio,
                    'Path_Diversity': np.round(final_path_diversity, 3),  
                }
                metrics_list.append(metrics)
                
                torch.cuda.empty_cache()
                
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_folder = os.path.join(base_output_folder, 'Generative_metric', self.args.exp_root_path)
            os.makedirs(metrics_folder, exist_ok=True)
            metrics_df.to_csv(os.path.join(metrics_folder, f"{self.args.sector}_Generative_metric.csv"), index=False)
            print(f"[INFO] 모든 Train 메트릭 저장 완료")
        else:
            print("[WARNING] 저장할 Train 메트릭 데이터가 없습니다.")
            
    def generate_fakes(self, log_returns_train, nz, cumsum=True, n=1, generator=None):
        """
        Using generators to generate virtual data (log return) and inverse scale (cumsumed real price)
        """
        if generator is None:
            generator = self.netG
                    
        fakes = []
        for i in range(n):
            noise = torch.randn(1, len(log_returns_train), nz, device=self.device)
            fake = generator(noise).detach().cpu().reshape(len(log_returns_train)).numpy()
            Ticker_fake = inverse(fake * self.log_returns_max, self.params) + self.log_returns_mean
            fakes.append(Ticker_fake)
            
        if n > 1:
            if not cumsum:
                return pd.DataFrame(fakes).T
            fakes_df = pd.DataFrame(fakes).T.cumsum()
            return fakes_df
        elif not cumsum:
            return Ticker_fake
        return Ticker_fake.cumsum()
    
    def extract_learned_noise(self, num_sequences, seq_length, nz, generator=None, device=None):
        """
        Extract noise samples using a generator.
        """
        if device is None:
            device = self.device
        if generator is None:
            generator = self.netG
        generator.eval()
        noise_samples = []
        with torch.no_grad():
            for _ in range(num_sequences):
                noise = torch.randn(1, seq_length, nz, device=device)
                fake = generator(noise).cpu().numpy().flatten()
                noise_samples.extend(fake)
        return np.array(noise_samples)
    
    def simulate_gbm_GAN(self, S0, mu, sigma, T, dt, nz, num_simulations, generator=None, device=None):
        """
        Simulate multiple GBM paths using the extracted noise.
        """
        if device is None:
            device = self.device
            
        if generator is None:
            generator = self.netG
            
        N = int(T / dt)
        S = np.zeros((num_simulations, N))
        S[:, 0] = S0
        learned_noises = self.extract_learned_noise(num_sequences=num_simulations, seq_length=N-1, nz=nz, generator=generator, device=device)
        
        learned_noises = learned_noises.reshape(num_simulations, N-1)
        
        learned_noises = (learned_noises - np.mean(learned_noises)) / np.std(learned_noises)
        
        for t in range(1, N):
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * learned_noises[:, t-1] * np.sqrt(dt))
        return S
    
    def simulate_gbm(self, num_repeats=10):
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
    
        base_output_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
            
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
                # GAN Model load
                netG_folder = os.path.join(base_output_folder, f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}')
                netG_files = [f for f in os.listdir(netG_folder) if f.endswith('.pth') and 'netG' in f]
            
                target_window = f'Sliding_Window_{window_idx + 1}_netG'
                matching_files = [f for f in netG_files if target_window in f]
                
                if not matching_files:
                    print(f"[WARNING] No .pth file containing '{target_window}' found in {netG_folder}. Skipping window {window_idx+1} for {ticker_name}.")
                    continue
            
                netG_file = os.path.join(netG_folder, matching_files[0])
                
                print(f"[INFO] {ticker_name} - Window {window_idx+1}: 모델 로드 중: {netG_file}")
                self.netG = torch.load(netG_file, map_location=self.device)
                self.netG.eval()
                
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
                    
                    gbm_paths = self.simulate_gbm_GAN(
                        S0=S0,
                        mu=mu,
                        sigma=sigma,
                        T=T,
                        dt=dt,
                        nz=self.args.noise_input_size,
                        num_simulations=self.args.num_simulations,
                    )
                    
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
                    simulation_folder = os.path.join(base_output_folder, 'GBM_scenario_2', self.args.exp_root_path, self.args.sector, ticker_name)
                    os.makedirs(simulation_folder, exist_ok=True)
                    
                    plot_simulations(
                        model_name=self.args.model_name,
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
            print(f"[INFO] 모든 평가 지표 저장 완료")
        else:
            print("[WARNING] 저장할 메트릭 데이터가 없습니다.")
    
    def evaluate_gan(self):
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
                
        base_output_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
        
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
                print(f"\n--- Sliding Window {window_idx+1} ---")
                print(f"  Train period: {window_info['train_start']} ~ {window_info['train_end']} | Rows: {window_info['train_rows']}")
                print(f"  Test period: {window_info['test_start']} ~ {window_info['test_end']} | Rows: {window_info['test_rows']}")

                # Step 1: Data preprocessing (GMM)
                log_returns_train = np.log(train_df['Adj Close'] / train_df['Adj Close'].shift(1))[1:].values
                log_returns_mean, params, log_returns_max, log_processed = self._get_data(log_returns_train)
                self.log_returns_mean = log_returns_mean
                self.params = params
                self.log_returns_max = log_returns_max

                # Load pre-trained model
                output_folder = os.path.join(base_output_folder, f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/')
                model_files = [f for f in os.listdir(output_folder) if f.startswith(f"Sliding_Window_{window_idx+1}_netG") and f.endswith('.pth')]
                if not model_files:
                    print(f"[WARNING] No model file found for Sliding_Window_{window_idx+1} in {output_folder}. Skipping.")
                    continue
                model_save_path = os.path.join(output_folder, model_files[0])
                self.netG = torch.load(model_save_path, map_location=self.device)
                self.netG.eval()

                # KS-test와 MMD
                fake_data = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(log_returns_train), nz=self.args.noise_input_size, device=self.device)
                fake_logs = fake_data.flatten()
                real_logs = log_processed.flatten()
                
                final_ks_stat, final_ks_pvalue = ks_2samp(real_logs, fake_logs)
                final_mmd_score = self._calculate_mmd(real_logs, fake_logs, sigma=1.0)

                # Coverage Ratio와 Path Diversity (기존 방식: 한 번에 fake_sample 수만큼 생성)
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                
                fake_cumsum_data = self.generate_fakes(log_returns_train, nz=self.args.noise_input_size, cumsum=True, n=self.args.fake_sample)
                fake_array = P0 * np.exp(fake_cumsum_data.values)
                
                # Calculate coverage ratio for 95% confidence interval
                conf_level = 0.95
                alpha = (1 - conf_level) / 2 * 100
                lower_bound = np.percentile(fake_array, alpha, axis=1)
                upper_bound = np.percentile(fake_array, 100 - alpha, axis=1)
                
                within_bounds = (real_prices >= lower_bound) & (real_prices <= upper_bound)
                final_coverage_ratio = np.mean(within_bounds)
                
                # Calculate path diversity
                final_path_diversity = calculate_path_diversity(fake_array)

                # Metrics 저장
                metrics = {
                    'Sector': self.args.sector,
                    'Ticker': ticker_name,
                    'Window': window_idx + 1,
                    'Train_Start': window_info['train_start'],
                    'Train_End': window_info['train_end'],
                    'KS_Stat': np.round(final_ks_stat, 3),
                    'KS_Pvalue': np.round(final_ks_pvalue, 3),
                    'MMD_Score': np.round(final_mmd_score, 3),
                    'Coverage_Ratio': np.round(final_coverage_ratio, 3),
                    'Path_Diversity': np.round(final_path_diversity, 3),
                }
                metrics_list.append(metrics)

                torch.cuda.empty_cache()

        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_folder = os.path.join(base_output_folder, 'Generative_metric_eval', self.args.exp_root_path)
            os.makedirs(metrics_folder, exist_ok=True)
            metrics_df.to_csv(os.path.join(metrics_folder, f"{self.args.sector}_Generative_metric_eval.csv"), index=False)
            print(f"[INFO] 모든 평가 메트릭 저장 완료")
        else:
            print("[WARNING] 저장할 평가 메트릭 데이터가 없습니다.")
            
            
    
    ''' def evaluate_gan(self):
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
                
        base_output_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
        
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
                print(f"\n--- Sliding Window {window_idx+1} ---")
                print(f"  Train period: {window_info['train_start']} ~ {window_info['train_end']} | Rows: {window_info['train_rows']}")
                print(f"  Test period: {window_info['test_start']} ~ {window_info['test_end']} | Rows: {window_info['test_rows']}")

                # Step 1: Data preprocessing (GMM)
                log_returns_train = np.log(train_df['Adj Close'] / train_df['Adj Close'].shift(1))[1:].values
                log_returns_mean, params, log_returns_max, log_processed = self._get_data(log_returns_train)
                self.log_returns_mean = log_returns_mean
                self.params = params
                self.log_returns_max = log_returns_max

                # Load pre-trained model
                output_folder = os.path.join(base_output_folder, f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/')
                model_files = [f for f in os.listdir(output_folder) if f.startswith(f"Sliding_Window_{window_idx+1}_netG") and f.endswith('.pth')]
                if not model_files:
                    print(f"[WARNING] No model file found for Sliding_Window_{window_idx+1} in {output_folder}. Skipping.")
                    continue
                model_save_path = os.path.join(output_folder, model_files[0])
                self.netG = torch.load(model_save_path, map_location=self.device)
                self.netG.eval()

                # KS-test와 MMD를 위한 반복 시행
                ks_stats = []
                ks_pvalues = []
                mmd_scores = []

                # fake_sample 수만큼 반복 (KS-test와 MMD만)
                for _ in range(self.args.fake_sample):
                    # Generate fake log returns samples
                    fake_data = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(log_returns_train), nz=self.args.noise_input_size, device=self.device)
                    fake_logs = fake_data.flatten()
                    real_logs = log_returns_train.flatten()
                    
                    # KS-test
                    ks_stat, ks_pvalue = ks_2samp(real_logs, fake_logs)
                    ks_stats.append(ks_stat)
                    ks_pvalues.append(ks_pvalue)
                    
                    # MMD
                    mmd_score = self._calculate_mmd(real_logs, fake_logs, sigma=1.0)
                    mmd_scores.append(mmd_score)

                # KS-test와 MMD 결과 집계 (평균)
                final_ks_stat = np.mean(ks_stats)
                final_ks_pvalue = np.mean(ks_pvalues)
                final_mmd_score = np.mean(mmd_scores)

                # Coverage Ratio와 Path Diversity (기존 방식: 한 번에 fake_sample 수만큼 생성)
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                
                fake_cumsum_data = self.generate_fakes(log_returns_train, nz=self.args.noise_input_size, cumsum=True, n=self.args.fake_sample)
                fake_array = P0 * np.exp(fake_cumsum_data.values)
                
                # Calculate coverage ratio for 95% confidence interval
                conf_level = 0.95
                alpha = (1 - conf_level) / 2 * 100
                lower_bound = np.percentile(fake_array, alpha, axis=1)
                upper_bound = np.percentile(fake_array, 100 - alpha, axis=1)
                
                within_bounds = (real_prices >= lower_bound) & (real_prices <= upper_bound)
                final_coverage_ratio = np.mean(within_bounds)
                
                # Calculate path diversity
                final_path_diversity = calculate_path_diversity(fake_array)

                # Metrics 저장
                metrics = {
                    'Sector': self.args.sector,
                    'Ticker': ticker_name,
                    'Window': window_idx + 1,
                    'Train_Start': window_info['train_start'],
                    'Train_End': window_info['train_end'],
                    'KS_Stat': np.round(final_ks_stat, 3),
                    'KS_Pvalue': np.round(final_ks_pvalue, 3),
                    'MMD_Score': np.round(final_mmd_score, 3),
                    'Coverage_Ratio': np.round(final_coverage_ratio, 3),
                    'Path_Diversity': np.round(final_path_diversity, 3),
                }
                metrics_list.append(metrics)

                torch.cuda.empty_cache()

        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_folder = os.path.join(base_output_folder, 'Generative_metric_eval', self.args.exp_root_path)
            os.makedirs(metrics_folder, exist_ok=True)
            metrics_df.to_csv(os.path.join(metrics_folder, f"{self.args.sector}_Generative_metric_eval.csv"), index=False)
            print(f"[INFO] 모든 평가 메트릭 저장 완료")
        else:
            print("[WARNING] 저장할 평가 메트릭 데이터가 없습니다.") '''