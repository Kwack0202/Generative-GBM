from common_imports import *

plt.rcParams.update({
    'axes.titlesize': 40,
    'axes.labelsize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 30
})

# 시뮬레이션 결과 시각화 함수 (출력 대신 PNG 파일로 저장)
def plot_simulations(model_name, real_prices, fake_prices, test_close, simulated_S, num_plot=10, save_path='plot_simulations.png'):
    plt.figure(figsize=(30, 15))
    
    # 학습 기간의 실제 데이터 플롯
    plt.plot(real_prices, color='blue', label='Train price', linewidth=4)
    
    # 학습 기간의 가상 데이터 플롯
    if fake_prices is not None and len(fake_prices) > 0:
        plt.plot(fake_prices, color='orange', label='Train Fake Data', linewidth=2)
    
    # 시뮬레이션된 경로 플롯 
    for i in range(num_plot):
        plt.plot(range(len(real_prices), len(real_prices) + simulated_S.shape[1]), 
                 simulated_S[i], alpha=0.5, linewidth=4, 
                 label='GBM-scenario' if i == 0 else "")
    
    # 테스트 기간의 실제 데이터 플롯
    ''' plt.plot(range(len(real_prices), len(real_prices) + len(test_close)), 
             test_close, color='red', label='Test price', linewidth=4) '''
    
    if model_name in ['standard_gbm', 'empirical_gbm']:
        plt.title(f'{model_name}')
    else:
        plt.title(f'{model_name}-GBM')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # 지정한 경로에 저장 (디렉토리가 없으면 생성)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def plot_confidence_interval(real_prices, bounds, test_close, simulated_S, num_plot=10, save_path='confidence_interval.png'):
    """
    Train 구간의 실제 가격과 GAN으로 생성된 데이터의 50%, 95%, 99% 신뢰구간을 시각화합니다.
    
    Args:
        real_prices (array-like): 실제 train 구간의 주가.
        bounds (dict): 신뢰 구간별 lower_bound와 upper_bound를 포함한 딕셔너리.
                       키는 신뢰 수준(0.5, 0.95, 0.99).
        test_close (array-like): 테스트 구간의 실제 주가.
        simulated_S (array-like): 시뮬레이션된 GBM 경로.
        num_plot (int): 플롯할 시뮬레이션 경로 수.
        save_path (str): 결과 플롯을 저장할 경로.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(real_prices, color='blue', label='Train Real Data', linewidth=1)
    
    # 50%, 95%, 99% 신뢰 구간 플롯
    colors = {'0.5': 'orange', '0.95': 'green', '0.99': 'purple'}
    for conf, bound in bounds.items():
        plt.fill_between(
            range(len(real_prices)),
            bound['lower'],
            bound['upper'],
            color=colors[str(conf)],
            alpha=0.3 - (conf - 0.5) * 0.1,  # 신뢰 수준이 높을수록 투명도 낮춤
            label=f'{int(conf*100)}% Confidence Interval'
        )
    
    # 시뮬레이션된 경로 플롯
    for i in range(num_plot):
        plt.plot(
            range(len(real_prices), len(real_prices) + simulated_S.shape[1]),
            simulated_S[i],
            alpha=0.5,
            linewidth=1,
            label='GBM-Data' if i == 0 else ""
        )
    
    # 테스트 기간의 실제 데이터 플롯
    plt.plot(
        range(len(real_prices), len(real_prices) + len(test_close)),
        test_close,
        color='red',
        label='Test Real Data',
        linewidth=1
    )
    
    plt.title('GenerativeAI-GBM')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close('all')
    
# 노이즈 분포 시각화 함수 (출력 대신 PNG 파일로 저장하고, K-S test 결과를 txt 파일로 저장)
def visualize_noise_distribution(real_noise, generated_noise, 
                                 hist_save_path='noise_distribution.png', 
                                 qq_save_path='qq_plot.png'):
    """
    실 노이즈와 생성된 노이즈의 분포를 커널 밀도 추정(KDE)과 Q-Q 플롯으로 저장합니다.
    K-S 테스트 결과는 여기서 저장하지 않습니다.
    """
    from scipy.stats import gaussian_kde
    import numpy as np
    
    # KDE 플롯 생성 및 저장
    plt.figure(figsize=(30, 16))
    
    # 실 노이즈 KDE
    kde_real = gaussian_kde(real_noise)
    x_real = np.linspace(min(real_noise), max(real_noise), 5000)
    plt.fill_between(x_real, kde_real(x_real), label='Real distribution', alpha=0.3, color='green')
    
    # 생성된 노이즈 KDE
    kde_gen = gaussian_kde(generated_noise)
    x_gen = np.linspace(min(generated_noise), max(generated_noise), 5000)
    plt.fill_between(x_gen, kde_gen(x_gen), label='Generated distribution', alpha=0.3, color='red')
    
    plt.legend()
    plt.title('Real vs. Generated distribution')
    plt.xlabel('Noise Value')
    plt.ylabel('Density')
    
    os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
    plt.savefig(hist_save_path)
    plt.close('all')

    # Q-Q Plot 생성 및 저장
    plt.figure(figsize=(30, 16))
    sm.qqplot(generated_noise, line='45', fit=True)
    plt.title('Q-Q Plot of Generated Noise')
    
    os.makedirs(os.path.dirname(qq_save_path), exist_ok=True)
    plt.savefig(qq_save_path)
    plt.close('all')