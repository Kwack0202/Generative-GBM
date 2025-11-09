from common_imports import *
from data.data_download import download_stock_data

from GBM.Mathematical_GBM import Exp_GBM
from GBM.exp.exp_gan import Exp_GAN
from GBM.exp.exp_diffusion import Exp_Diffusion

from stock_prediction.stock_prediction_GRU import Exp_STP_GRU
from stock_prediction.stock_prediction_TF import Exp_STP_TF
from stock_prediction.baseline_GRU import Exp_STP_Baseline_GRU
from stock_prediction.baseline_TF import Exp_STP_Baseline_TF

from stock_prediction.backtesting import Backtesting
from stock_prediction.portfolio import Portfolio
from stock_prediction.portfolio_strategy import Portfolio_strategy

from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os

fix_seed = 42
random.seed(fix_seed)
np.random.seed(fix_seed)

# 조합식 목록 정의
TRAIN_MONTHS_LIST = [12, 24, 36]
NUM_SCENARIOS_LIST = [25, 50, 75, 100]
SECTOR_LIST = [
    'Materials', 'ConsumerDiscretionary', 'ConsumerStaples', 'Energy', 'Financials',
    'Healthcare', 'Industrials', 'RealEstate', 'InformationTechnology', 'Utilities'
]

plt.rcParams.update({
    'axes.titlesize': 40,
    'axes.labelsize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 30
})

def run_experiment(args_dict):
    """단일 조합식 실행"""
    try:
        args = argparse.Namespace(**args_dict)
        print(f"[INFO] Running with task_name={args.task_name}, bt_name={args.bt_name}, train_months={args.train_months}, num_scenarios={args.num_scenarios}")
        
        if args.task_name == "Portfolio" and args.bt_name == "Portfolio_Construction":
            exp = Portfolio(args)
            exp.Portfolio_Construction()
        elif args.task_name == "Portfolio_strategy" and args.bt_name == "Portfolio_Construction":
            exp = Portfolio_strategy(args)
            exp.Portfolio_Construction()
        else:
            raise ValueError(f"Unsupported task_name='{args.task_name}' or bt_name='{args.bt_name}'")
        
        print(f"[INFO] Finished with train_months={args.train_months}, num_scenarios={args.num_scenarios}")
    except Exception as e:
        print(f"[ERROR] Failed for train_months={args_dict['train_months']}, num_scenarios={args_dict['num_scenarios']}: {str(e)}")


def main(args):
    if args.task_name == "data_download":
        print("Start downloading the original data")
        download_stock_data(
            args.metadata,
            args.exp_root_path,
            args.start_day, 
            args.end_day
        )
    
    elif args.task_name == 'Mathematical_GBM':
        print("Start simulation for all sliding windows using Standard GBM for test data")
        exp = Exp_GBM(args)
        exp.simulate_gbm(num_repeats=args.num_repeats)
    
    elif args.task_name == 'train':
        print("Start train the generativeAI")
        if args.model_type == 'GAN':
            setting = """
            task_name: {}
            exp_root_path: {}
            sector: {}
            model_type: {}
            model_name: {}
            test_start_year: {}
            total_test_months: {}
            sliding_test_months: {}
            train_months: {}
            noise_input_size: {}
            noise_output_size: {}
            model_optimizer: {}
            learning_rate: {}
            seq_len: {}
            batch_size: {}
            num_workers: {}
            num_epochs: {}
            min_epochs: {}
            check_interval: {}
            fake_sample: {}
            loss_tolerance: {}
            early_stop_patience: {}
            num_simulations:{}
            num_noise_samples:{}""".format(
                args.task_name,
                args.exp_root_path,
                args.sector,
                args.model_type,
                args.model_name,
                args.test_start_year,
                args.total_test_months,
                args.sliding_test_months,
                args.train_months,
                args.noise_input_size,
                args.noise_output_size,
                args.model_optimizer,
                args.learning_rate,
                args.seq_len,
                args.batch_size,
                args.num_workers,
                args.num_epochs,
                args.min_epochs,
                args.check_interval,
                args.fake_sample,
                args.loss_tolerance,
                args.early_stop_patience,
                args.num_simulations,
                args.num_noise_samples,
            )
            Exp = Exp_GAN
            exp = Exp(args)
            print("start training : {}".format(setting))
            exp.train_gan()
        
        elif args.model_type == 'Diffusion':
            setting = """
            task_name: {}
            exp_root_path: {}
            sector: {}
            model_type: {}
            model_name: {}
            test_start_year: {}
            total_test_months: {}
            sliding_test_months: {}
            train_months: {}
            noise_input_size: {}
            noise_output_size: {}
            model_optimizer: {}
            learning_rate: {}
            seq_len: {}
            batch_size: {}
            num_workers: {}
            num_epochs: {}
            min_epochs: {}
            check_interval: {}
            fake_sample: {}
            loss_tolerance: {}
            early_stop_patience: {}
            num_simulations:{}
            num_noise_samples:{}
            beta_start:{}
            beta_end:{}
            T_diffusion:{}""".format(
                args.task_name,
                args.exp_root_path,
                args.sector,
                args.model_type,
                args.model_name,
                args.test_start_year,
                args.total_test_months,
                args.sliding_test_months,
                args.train_months,
                args.noise_input_size,
                args.noise_output_size,
                args.model_optimizer,
                args.learning_rate,
                args.seq_len,
                args.batch_size,
                args.num_workers,
                args.num_epochs,
                args.min_epochs,
                args.check_interval,
                args.fake_sample,
                args.loss_tolerance,
                args.early_stop_patience,
                args.num_simulations,
                args.num_noise_samples,
                args.beta_start,
                args.beta_end,
                args.T_diffusion
            )
            Exp = Exp_Diffusion
            exp = Exp(args)
            print("start training : {}".format(setting))
            exp.train_diffusion()
    
    elif args.task_name == 'simulate':
        print("Start simulation for all sliding windows using GBM for test data")
        if args.model_type == 'GAN':
            exp = Exp_GAN(args)
            exp.simulate_gbm(num_repeats=args.num_repeats)
        elif args.model_type == 'Diffusion':
            exp = Exp_Diffusion(args)
            exp.simulate_gbm(num_repeats=args.num_repeats)
    
    elif args.task_name == 'sim':
        if args.model_type == 'GAN':
            exp = Exp_GAN(args)
            exp.evaluate_gan()
        elif args.model_type == 'Diffusion':
            exp = Exp_Diffusion(args)
            exp.evaluate_diffusion()    
        
        
    elif args.task_name == 'stock_prediction':
        exp = Exp_STP_TF(args)
        exp.process() 
    
    elif args.task_name == 'baseline_GRU':
        exp = Exp_STP_Baseline_GRU(args)
        exp.process()
    elif args.task_name == 'baseline_TF':
        exp = Exp_STP_Baseline_TF(args)
        exp.process()
    
    elif args.task_name == 'Backtesting':
        exp = Backtesting(args)
        if args.bt_name == 'UpDown_Signal':
            exp.UpDown_Signal()
        
        elif args.bt_name == 'BuySell_Signal':
            exp.BuySell_Signal()   
        elif args.bt_name == 'BuySell_Signal_YOY':
            exp.BuySell_Signal_YOY()
            
        elif args.bt_name == 'Simulation':
            exp.Simulation()  
        elif args.bt_name == 'Simulation_YOY':
            exp.Simulation_YOY()
            
        elif args.bt_name == 'BacktestSummary':
            exp.BacktestSummary()
        elif args.bt_name == 'BacktestSummary_YOY':
            exp.BacktestSummary_YOY()    
            
        elif args.bt_name == 'PlotResults':
            exp.PlotResults()
    
    elif args.task_name == 'Portfolio':
        exp = Portfolio(args)
        if args.bt_name == 'UpDown_Signal':
            exp.UpDown_Signal()
            
        elif args.bt_name == 'Portfolio_Trading':
            exp.Portfolio_Trading()
            
        elif args.bt_name == 'Portfolio_Construction':
            # sector 기본값 설정
            print(f"[DEBUG] args.train_months={args.train_months}, args.num_scenarios={args.num_scenarios}")
        
            # 조합식 생성
            combinations = list(product(
                TRAIN_MONTHS_LIST,
                NUM_SCENARIOS_LIST
            ))
            
            # 디버깅: 조합 수 확인
            print(f"[DEBUG] Total combinations: {len(combinations)}")
            for i, (tm, ns) in enumerate(combinations):
                print(f"[DEBUG] Combination {i+1}: train_months={tm}, num_scenarios={ns}")
            
            # 인자 딕셔너리 리스트 생성
            args_list = []
            for train_months, num_scenarios in combinations:
                args_dict = vars(args).copy()
                args_dict.update({
                    'train_months': train_months,
                    'num_scenarios': num_scenarios
                })
                args_list.append(args_dict)
            
            # 병렬 실행
            max_workers = os.cpu_count() -2  # I/O 병목 방지 위해 4개로 제한
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_experiment, args_dict) for args_dict in args_list]
                for future in tqdm(futures, desc="Processing combinations in parallel"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Exception in combination: {str(e)}")
        
        elif args.bt_name == 'Portfolio_Summary':
            exp.Portfolio_Summary()
                
    elif args.task_name == 'Portfolio_strategy':
        exp = Portfolio_strategy(args)
        
        if args.bt_name == 'Portfolio_Construction':
            # 디버깅: 입력 인자 출력
            print(f"[DEBUG] args.task_name={args.task_name}, args.bt_name={args.bt_name}, args.train_months={args.train_months}, args.num_scenarios={args.num_scenarios}")
            
            # 조합식 생성
            combinations = list(product(
                TRAIN_MONTHS_LIST,
                NUM_SCENARIOS_LIST
            ))
            
            # 디버깅: 조합 수 확인
            print(f"[DEBUG] Total combinations: {len(combinations)}")
            for i, (tm, ns) in enumerate(combinations):
                print(f"[DEBUG] Combination {i+1}: train_months={tm}, num_scenarios={ns}")
            
            # 인자 딕셔너리 리스트 생성
            args_list = []
            for train_months, num_scenarios in combinations:
                args_dict = vars(args).copy()
                args_dict.update({
                    'train_months': train_months,
                    'num_scenarios': num_scenarios
                })
                args_list.append(args_dict)
            
            # 병렬 처리
            max_workers = max(1, os.cpu_count() - 1)  # I/O 병목 방지
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_experiment, args_dict) for args_dict in args_list]
                for future in tqdm(futures, desc="Processing combinations in parallel"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Exception in combination: {str(e)}")
        
        elif args.bt_name == 'Portfolio_Summary':
            exp.Portfolio_Summary()

        elif args.bt_name == 'PlotResults':
            exp.PlotResults()
            
## ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock generative GBM")
    
    ## ==================================================
    ## basic config
    ## ==================================================
    parser.add_argument("--task_name", type=str, required=True, default="pretrain", help="task name [options : data_download]")
    
    ## ==================================================
    ## data download
    ## ==================================================

    parser.add_argument("--metadata", type=str, default='./datasets/metadata_info.csv', help="Path to the metadata CSV file")
    parser.add_argument('--exp_root_path', type=str, default='./datasets/SP500/', help='Data root for experiment')
    parser.add_argument('--start_day', type=str, default='2010-06-01', help='Start date (format : YYYY-MM-DD)') 
    parser.add_argument('--end_day', type=str, default='2025-01-10', help='End date (format : YYYY-MM-DD)')
    
    ## ==================================================
    ## GBM Mathematical modeling
    ## ==================================================
    parser.add_argument('--gbm_name', type=str, default='standard_gbm', help='Mathematical model name')
    parser.add_argument('--sector', type=str, default='', help='Data root for experiment')
    
    ## ==================================================
    ## GenerativeAI modeling
    ## ==================================================
    '''
    experiment data root & model define
    '''
    parser.add_argument('--model_type', type=str, default='GAN', help='generativeAI model type')
    parser.add_argument('--model_name', type=str, default='VanillaGAN', help='generativeAI model name')
    
    '''
    hyperparameter(sliding window)
    '''
    parser.add_argument('--test_start_year', type=int, default=2022, help='test_start_year')
    parser.add_argument('--total_test_months', type=int, default=36, help='total_test_months')
    parser.add_argument('--sliding_test_months', type=int, default=12, help='sliding_test_months')
    parser.add_argument('--train_months', type=int, default=36, help='train_months')
    
    '''
    hyperparameter(modeling)
    '''
    parser.add_argument('--noise_input_size', type=int, default=3, help='The noise vector dimension of the generator(input)')
    parser.add_argument('--noise_output_size', type=int, default=1, help='The noise vector dimension of the generator(output)')
    parser.add_argument('--model_optimizer', type=str, default='Adam', help='model_optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning_rate')
    parser.add_argument('--seq_len', type=int, default=127, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--num_epochs', type=int, default=2000, help='num_epochs')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta_start')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta_end')
    parser.add_argument('--T_diffusion', type=int, default=1000, help='diffusion steps')
    
    '''
    hyperparameter(monitoring)
    '''
    parser.add_argument('--min_epochs', type=int, default=500, help='min_epochs')
    parser.add_argument('--check_interval', type=int, default=10, help='check_interval')
    parser.add_argument('--ks_threshold', type=float, default=0.05, help='ks_threshold')
    parser.add_argument('--pvalue_threshold', type=float, default=0.05, help='ks_pvalue_threshold')
    parser.add_argument('--fid_threshold', type=float, default=0.5, help='fid_threshold')
    parser.add_argument('--mmd_threshold', type=float, default=0.5, help='mmd_threshold')
    parser.add_argument('--fake_sample', type=int, default=10, help='fake_sample')
    parser.add_argument('--confidence', type=float, default=0.8, help='confidence')
    parser.add_argument('--loss_tolerance', type=float, default=0.0001, help='loss_tolerance')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='early_stop_patience')
    
    '''
    hyperparameter(simulate)
    '''
    parser.add_argument('--num_simulations', type=int, default=10, help='num_simulations')
    parser.add_argument('--num_noise_samples', type=int, default=1, help='num_noise_samples')
    parser.add_argument('--num_repeats', type=int, default=10, help='Number of cross-validation repeats')
    
    '''
    stock prediction
    '''
    parser.add_argument('--GBM_type', type=str, default='./Mathematical/standard_gbm', help='Data root for experiment')
    parser.add_argument('--num_scenarios', type=int, default=100, help='Number of scenarios to use')
    
    '''
    backtesting
    '''
    parser.add_argument('--bt_name', type=str, default='pred_results', help='merge the pred results with origin stock data')
    parser.add_argument('--pf_name', type=str, default='PortfolioSimulation', help='merge the pred results with origin stock data')
    
    args = parser.parse_args()
    main(args)