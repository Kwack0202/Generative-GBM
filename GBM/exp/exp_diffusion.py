from common_imports import *
from GBM.exp.exp_basic import Exp_Basic
from GBM.exp.data_loader import StockDataset

from GBM.models.DDPM import DiffusionModel, LatentDiffusionModel

from utils.sliding_window import get_sliding_window_data
from utils.gmm_preprecessing import *
from utils.visualization import *
from utils.metric import *
import copy

np.seterr(all='warn')  # 모든 연산에 대해 경고 발생하도록 설정
warnings.filterwarnings('ignore', category=RuntimeWarning) 

class Exp_Diffusion(Exp_Basic):
    def __init__(self, args):
        super(Exp_Diffusion, self).__init__(args)
        self.model = None
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
    
    def _build_diffusion_model(self):
        if self.args.model_name == 'DDPM':
            model = DiffusionModel(
                T_diffusion=self.args.T_diffusion).to(self.device)
        
        elif self.args.model_name == 'LDM':
            model = LatentDiffusionModel(
                T_diffusion=self.args.T_diffusion,
                time_embed_dim=64,   
                hidden_dim=128,      
                latent_dim=64,      
                num_heads=4        
            ).to(self.device)
                        
        return model

    def _select_optimizer(self, model):
        if self.args.model_optimizer == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.args.learning_rate,
                betas=(0.5, 0.999),
                weight_decay=getattr(self.args, 'weight_decay', 1e-5)
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
            return optimizer, scheduler
        
        elif self.args.model_optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.args.learning_rate)
            return optimizer
        else:
            raise ValueError("Unknown optimizer specified")

    def _save_settings(self):
        settings_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
        os.makedirs(settings_folder, exist_ok=True)
        settings_path = os.path.join(settings_folder, f'settings.txt')
        with open(settings_path, 'w') as f:
            for key, value in vars(self.args).items():
                f.write(f"{key}: {value}\n")

    def _calculate_mmd(self, real_data, fake_data, sigma=1.0):
        real_data = np.asarray(real_data).reshape(-1, 1)
        fake_data = np.asarray(fake_data).reshape(-1, 1)
        n, m = len(real_data), len(fake_data)
        xx = np.exp(-cdist(real_data, real_data, metric='sqeuclidean') / (2 * sigma**2))
        yy = np.exp(-cdist(fake_data, fake_data, metric='sqeuclidean') / (2 * sigma**2))
        xy = np.exp(-cdist(real_data, fake_data, metric='sqeuclidean') / (2 * sigma**2))
        mmd = np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
        mmd = np.sqrt(max(mmd, 0))
        return np.round(mmd, 3)

    def _plot_noisy_data(self, x0, x_t, t, ticker_name, window_idx, output_folder):
        """Visualize and save noisy data at a specific timestep for both flattened and first sequence."""
        T_diffusion = self.args.T_diffusion
        t_percentage = (t / T_diffusion) * 100
        
        # 1. Flattened 데이터 시각화
        plt.figure(figsize=(20, 16))
        x0_np_flat = x0.cpu().numpy().flatten()
        x_t_np_flat = x_t.cpu().numpy().flatten()
        print(f"[DEBUG] Flattened x0_np length: {len(x0_np_flat)}, x_t_np length: {len(x_t_np_flat)}")
        plt.plot(x0_np_flat, label='Original Data (0%)', alpha=0.7)
        plt.plot(x_t_np_flat, label=f'Noisy Data ({t_percentage:.1f}%)', alpha=0.7)
        plt.title(f'Noisy Data : {t_percentage:.1f}%')
        plt.xlabel('Time Index')
        plt.ylabel('Normalized Log Returns')
        plt.legend()
        plt.grid(True)
        save_path_flat = os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_noisy_data_t_{t_percentage:.1f}%_flattened.png")
        plt.savefig(save_path_flat, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Flattened noisy data visualization saved at: {save_path_flat}")

        # 2. 첫 번째 시퀀스 시각화
        plt.figure(figsize=(20, 16))
        x0_np_seq = x0[0].cpu().numpy().flatten()  # 첫 번째 시퀀스만 선택
        x_t_np_seq = x_t[0].cpu().numpy().flatten()  # 첫 번째 시퀀스만 선택
        print(f"[DEBUG] First sequence x0_np length: {len(x0_np_seq)}, x_t_np length: {len(x_t_np_seq)}")
        plt.plot(x0_np_seq, label='Original Data (0%)', alpha=0.7, linewidth=4)
        plt.plot(x_t_np_seq, label=f'Noisy Data ({t_percentage:.1f}%)', alpha=0.7, linewidth=4)
        plt.title(f'Noisy Data : {t_percentage:.1f}%')
        plt.xlabel('Time Index')
        plt.ylabel('Normalized Log Returns')
        plt.legend()
        plt.grid(True)
        save_path_seq = os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_noisy_data_t_{t_percentage:.1f}%_first_sequence.png")
        plt.savefig(save_path_seq, bbox_inches='tight')
        plt.close()
        print(f"[INFO] First sequence noisy data visualization saved at: {save_path_seq}")
          
    def train_diffusion(self):
        self._save_settings()
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
        
        base_output_folder = os.path.join(
            './outputs', self.args.model_type, self.args.model_name,
            f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/')
        
        metrics_list = []

        # Diffusion 하이퍼파라미터 정의
        T_diffusion = self.args.T_diffusion 
        beta_start = self.args.beta_start 
        beta_end = self.args.beta_end
        betas = torch.linspace(beta_start, beta_end, T_diffusion).to(self.device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0).to(self.device)

        for stock_file in stock_files:
            ticker_name = stock_file[:-4]
            print(f'\nDiffusion-GBM start ~~ Stock code: {ticker_name}')
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

                # Step 2: Model define
                model = self._build_diffusion_model()
                optimizer, scheduler = self._select_optimizer(model)
                mse_loss = nn.MSELoss()

                # Step 3: Dataset load
                dataset = StockDataset(log_processed, self.args.seq_len)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.args.num_workers)

                # Step 4: Model train
                progress = tqdm(range(self.args.num_epochs))
                
                best_loss = float('inf')
                early_stop_counter = 0     
                best_epoch = 0
                
                best_model = None
                best_ks_stat = None
                best_mmd_score = None
                
                prev_loss = float('inf')
                
                model.train()
                for epoch in progress:
                    for batch_idx, batch in enumerate(dataloader):
                        # batch: (batch_size, seq_len, 1) -> 마지막 차원 squeeze
                        x0 = batch.squeeze(-1).to(self.device)  # 원본(clean) 데이터
                        
                        # 각 배치에 대해 무작위 t (0 ~ T_diffusion-1) 선택
                        t_batch = torch.randint(0, T_diffusion, (x0.size(0),), device=self.device).long()
                        
                        # 각 t에 해당하는 sqrt(alpha_bar) 및 sqrt(1 - alpha_bar)
                        sqrt_alpha_bar = torch.sqrt(alpha_bars[t_batch]).to(self.device).view(-1, 1)
                        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t_batch]).to(self.device).view(-1, 1)
                        
                        # 노이즈 샘플: x0와 같은 shape
                        noise = torch.randn_like(x0)
                        # forward process: x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
                        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
                        
                        # Visualize noisy data at specific timesteps (first batch of first epoch)
                        if epoch == 0 and batch_idx == 0:
                            timesteps_to_visualize = [0, T_diffusion//4, T_diffusion//2, 3*T_diffusion//4, T_diffusion-1]
                            for t_vis in timesteps_to_visualize:
                                t_tensor = torch.full((x0.size(0),), t_vis, device=self.device).long()
                                sqrt_alpha_bar_vis = torch.sqrt(alpha_bars[t_tensor]).to(self.device).view(-1, 1)
                                sqrt_one_minus_alpha_bar_vis = torch.sqrt(1 - alpha_bars[t_tensor]).to(self.device).view(-1, 1)
                                noise_vis = torch.randn_like(x0)
                                x_t_vis = sqrt_alpha_bar_vis * x0 + sqrt_one_minus_alpha_bar_vis * noise_vis
                                output_folder = os.path.join(base_output_folder, 'noisy_plt',f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/')
                                os.makedirs(output_folder, exist_ok=True)
                                self._plot_noisy_data(x0, x_t_vis, t_vis, ticker_name, window_idx, output_folder)
                        
                        # 모델은 x_t와 t_batch를 입력받아 추가된 노이즈를 예측함.
                        noise_pred = model(x_t, t_batch)
                        
                        loss = mse_loss(noise_pred, noise)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    # Generate fake data for coverage ratio calculation
                    generated_noise = self.extract_learned_noise(num_sequences=self.args.num_noise_samples, seq_length=len(log_returns_train), device=self.device, model=model)
                    real_noise = log_processed.flatten()
                    
                    # Calculate KS statistic and MMD score
                    ks_stat, ks_pvalue = ks_2samp(real_noise, generated_noise)
                    mmd_score = self._calculate_mmd(real_noise, generated_noise, sigma=1.0)
                    
                    # progress description 
                    msg = (f'Loss: {loss.item():.8f} | KS statistic: {ks_stat:.6f} (p-value: {ks_pvalue:.4f}) | MMD Score: {mmd_score:.3f}')
                    progress.set_description(msg)

                    # Monitoring & EarlyStop
                    if (epoch + 1) >= self.args.min_epochs:
                        if (loss.item() < prev_loss - self.args.loss_tolerance and
                            ks_stat < 0.05 and
                            ks_pvalue > 0.05 and
                            mmd_score < 0.05):
                            best_loss = loss.item()
                            best_epoch = epoch + 1
                            best_model = copy.deepcopy(model)
                            best_ks_stat = ks_stat
                            best_mmd_score = mmd_score
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if (early_stop_counter >= self.args.early_stop_patience and
                                ks_stat < 0.05 and
                                ks_pvalue > 0.05 and
                                mmd_score < 0.05):
                                print(f"\n[Early Stopping] Epoch {epoch+1}: No improvement for {early_stop_counter} consecutive epochs.")
                                break

                    prev_loss = loss.item()

                if best_model is not None:
                    print(f"\n[Best INFO] Epoch: {best_epoch} | Loss: {best_loss:.8f} | KS Statistic: {best_ks_stat:.6f} | MMD Score: {best_mmd_score:.3f}")
                    model = best_model
                    final_epoch = best_epoch
                else:
                    final_epoch = epoch + 1

                output_folder = os.path.join(base_output_folder, f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/')
                os.makedirs(output_folder, exist_ok=True)
                
                model_save_path = os.path.join(output_folder, f"Sliding_Window_{window_idx+1}_diffusion_epoch_{final_epoch}.pth")
                torch.save(model.state_dict(), model_save_path)

                # Step 5: Simulation
                self.model = self._build_diffusion_model()
                self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
                self.model.eval()

                '''
                Generate Fake Data using GAN (Train period)
                '''
                real_cumsum = log_returns_train.cumsum()
                fakes_cumsum = self.generate_fakes(log_returns_train, cumsum=True).flatten()
                P0 = train_df['Adj Close'].iloc[0]
                real_prices = P0 * np.exp(real_cumsum)
                fake_prices = P0 * np.exp(fakes_cumsum)

                '''
                Generate Confidence coverage using GAN (Train period)
                '''
                fake_data = self.generate_fakes(log_returns_train, cumsum=True, n=self.args.fake_sample)
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
                
                gbm_paths = self.simulate_gbm_diffusion(
                    S0=S0_price,
                    mu=np.mean(log_returns_train) * 252,
                    sigma=np.std(log_returns_train) * np.sqrt(252),
                    T=T,
                    dt=dt,
                    num_simulations=self.args.num_simulations,
                    device=self.device
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

                learned_noise = self.extract_learned_noise(num_sequences=self.args.num_noise_samples, seq_length=len(log_returns_train), device=self.device)
                real_noise = log_processed.flatten()
                
                final_ks_stat, final_ks_pvalue = ks_2samp(real_noise, learned_noise)
                final_mmd_score = self._calculate_mmd(real_noise, learned_noise, sigma=1.0)

                # Calculate coverage ratio for 95% confidence interval
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                
                fake_data = self.generate_fakes(log_returns_train, cumsum=True, n=self.args.fake_sample, model=model)
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

    def _sample_diffusion(self, sample_shape, model, device):
        T_diffusion = self.args.T_diffusion if hasattr(self.args, 'T_diffusion') else 200
        beta_start = self.args.beta_start if hasattr(self.args, 'beta_start') else 1e-4
        beta_end = self.args.beta_end if hasattr(self.args, 'beta_end') else 0.02
        betas = torch.linspace(beta_start, beta_end, T_diffusion).to(device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0).to(device)

        model.eval()
        with torch.no_grad():
            x = torch.randn(sample_shape, device=device)
            for t_step in reversed(range(T_diffusion)):
                t_tensor = torch.full((sample_shape[0],), t_step, device=device, dtype=torch.long)
                beta_t = betas[t_step].to(device)
                alpha_t = alphas[t_step].to(device)
                alpha_bar_t = alpha_bars[t_step].to(device)
                epsilon_theta = model(x, t_tensor)
                x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)
                if t_step > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
            return x

    def generate_fakes(self, log_returns_train, cumsum=True, n=1, model=None):
        if model is None:
            model = self.model

        model.eval()
        with torch.no_grad():
            fake = self._sample_diffusion((n, len(log_returns_train)), model, self.device)
            fake = fake.cpu().numpy()
            if n > 1:
                fake = fake.reshape(n, -1)
                Ticker_fake = np.array([inverse(f * self.log_returns_max, self.params) + self.log_returns_mean for f in fake])
                if cumsum:
                    return pd.DataFrame(Ticker_fake).T.cumsum()
                return pd.DataFrame(Ticker_fake).T
            else:
                fake = fake.flatten()
                Ticker_fake = inverse(fake * self.log_returns_max, self.params) + self.log_returns_mean
                if cumsum:
                    return Ticker_fake.cumsum()
                return Ticker_fake

    def extract_learned_noise(self, num_sequences, seq_length, device, model=None):
        if model is None:
            model = self.model

        model.eval()
        noise_samples = []
        with torch.no_grad():
            for _ in range(num_sequences):
                fake = self._sample_diffusion((1, seq_length), model, device)
                noise_samples.extend(fake.cpu().numpy().flatten())
        return np.array(noise_samples)

    def simulate_gbm_diffusion(self, S0, mu, sigma, T, dt, num_simulations, device, model=None):
        if model is None:
            model = self.model

        N = int(T / dt)
        S = np.zeros((num_simulations, N))
        S[:, 0] = S0
        learned_noises = []
        for _ in range(num_simulations):
            noise_seq = self._sample_diffusion((1, N-1), model, device)
            learned_noises.append(noise_seq.cpu().numpy().flatten())
        
        learned_noises = np.array(learned_noises)
        
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
        
        T_diffusion = self.args.T_diffusion
        beta_start = self.args.beta_start
        beta_end = self.args.beta_end
        betas = torch.linspace(beta_start, beta_end, T_diffusion).to(self.device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0).to(self.device)

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
                # Diffusion Model load
                
                model_folder = os.path.join(base_output_folder, f'{self.args.exp_root_path}/{self.args.sector}/{ticker_name}')
                model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth') and 'diffusion' in f]
                
                target_window = f'Sliding_Window_{window_idx + 1}_diffusion'
                matching_files = [f for f in model_files if target_window in f]
                
                if not matching_files:
                    print(f"[WARNING] No .pth file containing '{target_window}' found in {model_folder}. Skipping window {window_idx+1} for {ticker_name}.")
                    continue
                
                model_file = os.path.join(model_folder, matching_files[0])
                                
                print(f"[INFO] {ticker_name} - Window {window_idx+1}: 모델 로드 중: {model_file}")
                self.model = self._build_diffusion_model()
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.eval()

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
                    
                    gbm_paths = self.simulate_gbm_diffusion(
                        S0=S0,
                        mu=mu,
                        sigma=sigma,
                        T=T,
                        dt=dt,
                        num_simulations=self.args.num_simulations,
                        device=self.device
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
                    simulation_folder = os.path.join(base_output_folder, 'GBM_scenario_4', self.args.exp_root_path, self.args.sector, ticker_name)
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
            
    def evaluate_diffusion(self):
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
                model_files = [f for f in os.listdir(output_folder) if f.startswith(f"Sliding_Window_{window_idx+1}_diffusion") and f.endswith('.pth')]
                if not model_files:
                    print(f"[WARNING] No model file found for Sliding_Window_{window_idx+1} in {output_folder}. Skipping.")
                    continue
                model_save_path = os.path.join(output_folder, model_files[0])
                self.model = self._build_diffusion_model()
                self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
                self.model.eval()

                # Generate learned noise and calculate metrics
                learned_noise = self.extract_learned_noise(num_sequences=self.args.num_noise_samples, seq_length=len(log_returns_train), device=self.device)
                real_noise = log_processed.flatten()
                
                final_ks_stat, final_ks_pvalue = ks_2samp(real_noise, learned_noise)
                final_mmd_score = self._calculate_mmd(real_noise, learned_noise, sigma=1.0)

                # Calculate coverage ratio for 95% confidence interval
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                
                fake_data = self.generate_fakes(log_returns_train, cumsum=True, n=self.args.fake_sample)
                fake_array = P0 * np.exp(fake_data.values)
                
                conf_level = 0.95
                alpha = (1 - conf_level) / 2 * 100
                lower_bound = np.percentile(fake_array, alpha, axis=1)
                upper_bound = np.percentile(fake_array, 100 - alpha, axis=1)
                
                within_bounds = (real_prices >= lower_bound) & (real_prices <= upper_bound)
                final_coverage_ratio = np.mean(within_bounds)

                # Calculate path diversity
                final_path_diversity = calculate_path_diversity(fake_array)

                metrics = {
                    'Sector': self.args.sector,
                    'Ticker': ticker_name,
                    'Window': window_idx + 1,
                    'Train_Start': window_info['train_start'],
                    'Train_End': window_info['train_end'],
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
            metrics_folder = os.path.join(base_output_folder, 'Generative_metric_eval', self.args.exp_root_path)
            os.makedirs(metrics_folder, exist_ok=True)
            metrics_df.to_csv(os.path.join(metrics_folder, f"{self.args.sector}_Generative_metric_eval.csv"), index=False)
            print(f"[INFO] 모든 평가 메트릭 저장 완료")
        else:
            print("[WARNING] 저장할 평가 메트릭 데이터가 없습니다.")
            
            
    ''' def evaluate_diffusion(self):
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
                model_files = [f for f in os.listdir(output_folder) if f.startswith(f"Sliding_Window_{window_idx+1}_diffusion") and f.endswith('.pth')]
                if not model_files:
                    print(f"[WARNING] No model file found for Sliding_Window_{window_idx+1} in {output_folder}. Skipping.")
                    continue
                model_save_path = os.path.join(output_folder, model_files[0])
                self.model = self._build_diffusion_model()
                self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
                self.model.eval()

                # KS-test와 MMD를 위한 반복 시행
                ks_stats = []
                ks_pvalues = []
                mmd_scores = []

                # fake_sample 수만큼 반복 (KS-test와 MMD만)
                for _ in range(self.args.fake_sample):
                    # Generate learned noise
                    learned_noise = self.extract_learned_noise(num_sequences=self.args.num_noise_samples, seq_length=len(log_returns_train), device=self.device)
                    real_noise = log_processed.flatten()
                    
                    # KS-test
                    ks_stat, ks_pvalue = ks_2samp(real_noise, learned_noise)
                    ks_stats.append(ks_stat)
                    ks_pvalues.append(ks_pvalue)
                    
                    # MMD
                    mmd_score = self._calculate_mmd(real_noise, learned_noise, sigma=1.0)
                    mmd_scores.append(mmd_score)

                # KS-test와 MMD 결과 집계 (평균)
                final_ks_stat = np.mean(ks_stats)
                final_ks_pvalue = np.mean(ks_pvalues)
                final_mmd_score = np.mean(mmd_scores)

                # Coverage Ratio와 Path Diversity (기존 방식: 한 번에 fake_sample 수만큼 생성)
                P0 = train_df['Adj Close'].iloc[0]
                real_cumsum = log_returns_train.cumsum()
                real_prices = P0 * np.exp(real_cumsum)
                
                fake_data = self.generate_fakes(log_returns_train, cumsum=True, n=self.args.fake_sample)  # 기존처럼 fake_sample 수만큼 생성
                fake_array = P0 * np.exp(fake_data.values)
                
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