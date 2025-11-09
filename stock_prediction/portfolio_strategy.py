from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.visualization import *
from utils.metric import *

class Portfolio_strategy:
    def __init__(self, args):
        self.args = args
        print(f"[INFO] Current working directory: {os.getcwd()}")
        # 메타데이터 로드
        metadata_path = "./datasets/metadata_info.csv"
        print(f"[INFO] Attempting to load metadata from: {metadata_path}")
        try:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")
            self.meta_df = pd.read_csv(metadata_path)
            self.tickers = self.meta_df['Ticker'].tolist()
            if len(self.tickers) != 150:
                print(f"[WARNING] Expected 150 tickers in metadata, found {len(self.tickers)}")
            else:
                print(f"[INFO] Successfully loaded {len(self.tickers)} tickers from metadata")
        except Exception as e:
            print(f"[ERROR] Failed to load metadata: {str(e)}")
            self.meta_df = None
            self.tickers = []

    def Portfolio_Construction(self):
        # 기본 경로 설정
        base_input_path = "./stock_prediction/individual_trading/simulation/GRU_13_64_2/"
        base_output_path = "./stock_prediction/portfolio_strategy/portfolio_results/GRU_13_64_2/"
        model_name = "GRU_13_64_2"
        
        # 거래 설정
        initial_seed_per_ticker = 1000  # 종목당 초기 시드
        commission_rate = 0.0005  # 수수료 0.05%
        
        # 단일 조합 처리
        train_months = self.args.train_months
        num_scenarios = self.args.num_scenarios
        time_series_length = f"Train_{train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
        
        for model in tqdm(model_types, desc=f"Processing models for train_months={train_months}, num_scenarios={num_scenarios}"):
            # 입력/출력 경로 설정
            method_input_path = os.path.join(base_input_path, model)
            if model == "baseline":
                input_base_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path)
                output_base_path = os.path.join(base_output_path, model, time_series_length, self.args.exp_root_path)
                scenarios = 0
            else:
                input_base_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{num_scenarios}", self.args.exp_root_path)
                output_base_path = os.path.join(base_output_path, model, time_series_length, f"Scenarios_{num_scenarios}", self.args.exp_root_path)
                scenarios = num_scenarios
            
            if not os.path.exists(input_base_path):
                print(f"[WARNING] Input path {input_base_path} does not exist. Skipping.")
                continue
            
            # 포트폴리오 유형 정의
            portfolio_types = {
                'all_tickers': {'key': 'all', 'tickers': self.tickers},
                'sector': {'key': 'Sector', 'values': self.meta_df['Sector'].unique()},
                'risk_category': {'key': 'Risk_Category', 'values': self.meta_df['Risk_Category'].unique()},
                'market_cap': {'key': 'Market Cap Classification', 'values': self.meta_df['Market Cap Classification'].unique()}
            }
            
            for portfolio_type, config in portfolio_types.items():
                if portfolio_type == 'all_tickers':
                    portfolio_keys = [config['key']]
                else:
                    portfolio_keys = config['values']
                
                for portfolio_key in portfolio_keys:
                    # 포트폴리오 티커 선별
                    if portfolio_type == 'all_tickers':
                        selected_tickers = self.tickers
                    else:
                        selected_tickers = self.meta_df[self.meta_df[config['key']] == portfolio_key]['Ticker'].tolist()
                    
                    if not selected_tickers:
                        print(f"[WARNING] No tickers found for {portfolio_type}/{portfolio_key}. Skipping.")
                        continue
                    
                    # 포트폴리오 결과 저장용 데이터프레임
                    portfolio_data = {
                        'index': [],  # 날짜
                        'Margin_Profit': [],
                        'Margin_Return': [],
                        'Cumulative_Profit': [],
                        'Cumulative_Return': [],
                        'Drawdown': [],
                        'Drawdown_rate': [],
                        'Holding_Tickers': []
                    }
                    
                    # 각 티커별 잔액 추적
                    ticker_balances = {ticker: initial_seed_per_ticker for ticker in selected_tickers}
                    cumulative_profit = 0
                    peak_profit = 0
                    peak_return = 0
                    
                    # 날짜별 데이터 수집
                    date_index = None
                    ticker_data = {}
                    for ticker in selected_tickers:
                        ticker_sector = self.meta_df[self.meta_df['Ticker'] == ticker]['Sector'].iloc[0]
                        signal_file = os.path.join(input_base_path, ticker_sector, f"{ticker}.csv")
                        if not os.path.exists(signal_file):
                            print(f"[WARNING] Signal file {signal_file} does not exist. Skipping {ticker}.")
                            continue
                        
                        df = pd.read_csv(signal_file)
                        ticker_data[ticker] = df
                        if date_index is None:
                            date_index = df['index'].tolist()
                    
                    if not date_index:
                        print(f"[WARNING] No valid data for {portfolio_type}/{portfolio_key}. Skipping.")
                        continue
                    
                    # 날짜별 포트폴리오 계산
                    for i, date in enumerate(date_index):
                        daily_profit = 0
                        daily_return = 0
                        holding_tickers = []
                        
                        for ticker in selected_tickers:
                            if ticker not in ticker_data:
                                continue
                            
                            df = ticker_data[ticker]
                            if i >= len(df):
                                continue
                            
                            row = df.iloc[i]
                            if row['action'] in ['Buy', 'Sell']:
                                holding_tickers.append(ticker)
                            
                            if row['Margin_Profit'] != 0:
                                # 거래 발생: 잔액 업데이트
                                profit = row['Margin_Profit'] * (ticker_balances[ticker] / initial_seed_per_ticker)
                                ticker_balances[ticker] += profit
                                daily_profit += profit
                                daily_return += row['Margin_Return'] * (ticker_balances[ticker] / sum(ticker_balances.values()))
                        
                        cumulative_profit += daily_profit
                        cumulative_return = (cumulative_profit / (initial_seed_per_ticker * len(selected_tickers))) * 100
                        
                        # 드로다운 계산
                        if cumulative_profit > peak_profit:
                            peak_profit = cumulative_profit
                        drawdown = -(peak_profit - cumulative_profit)
                        
                        if cumulative_return > peak_return:
                            peak_return = cumulative_return
                        drawdown_rate = -(peak_return - cumulative_return)
                        
                        # 결과 저장
                        portfolio_data['index'].append(date)
                        portfolio_data['Margin_Profit'].append(daily_profit)
                        portfolio_data['Margin_Return'].append(daily_return)
                        portfolio_data['Cumulative_Profit'].append(cumulative_profit)
                        portfolio_data['Cumulative_Return'].append(cumulative_return)
                        portfolio_data['Drawdown'].append(drawdown)
                        portfolio_data['Drawdown_rate'].append(drawdown_rate)
                        portfolio_data['Holding_Tickers'].append(','.join(holding_tickers))
                    
                    # 데이터프레임 생성
                    portfolio_df = pd.DataFrame(portfolio_data)
                    portfolio_df = portfolio_df.round(3)
                    
                    # 출력 저장
                    output_dir = os.path.join(output_base_path, portfolio_type)
                    output_file = os.path.join(output_dir, f"{portfolio_key.replace(' ', '_')}.csv")
                    os.makedirs(output_dir, exist_ok=True)
                    portfolio_df.to_csv(output_file, index=False)
                    print(f"[INFO] Saved portfolio results for {portfolio_type}/{portfolio_key} to {output_file}")

    def Portfolio_Summary(self):
        base_input_path = "./stock_prediction/portfolio_strategy/portfolio_results/GRU_13_64_2/"
        base_output_path = "./stock_prediction/portfolio_strategy/portfolio_backtesting/"
        model_name = "GRU_13_64_2"
        
        # 실험 파라미터
        TRAIN_MONTHS_LIST = [12, 24, 36]
        NUM_SCENARIOS_LIST = [25, 50, 75, 100]
        portfolio_types = ['all_tickers', 'sector', 'risk_category', 'market_cap']
        
        summary_data = []
        
        # 모든 조합식 순회
        for train_months in tqdm(TRAIN_MONTHS_LIST, desc="Processing train months"):
            for num_scenarios in NUM_SCENARIOS_LIST:
                time_series_length = f"Train_{train_months}_Test_{self.args.sliding_test_months}"
                
                model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
                
                for model in model_types:
                    method_input_path = os.path.join(base_input_path, model)
                    if model == "baseline":
                        input_base_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path)
                        scenarios = 0
                    else:
                        input_base_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{num_scenarios}", self.args.exp_root_path)
                        scenarios = num_scenarios
                    
                    if not os.path.exists(input_base_path):
                        print(f"[WARNING] Input path {input_base_path} does not exist. Skipping.")
                        continue
                    
                    for portfolio_type in portfolio_types:
                        portfolio_path = os.path.join(input_base_path, portfolio_type)
                        if not os.path.exists(portfolio_path):
                            print(f"[WARNING] Portfolio path {portfolio_path} does not exist. Skipping.")
                            continue
                        
                        portfolio_keys = [f[:-4] for f in os.listdir(portfolio_path) if f.endswith('.csv')]
                        if not portfolio_keys:
                            print(f"[WARNING] No portfolio files found in {portfolio_path}. Skipping.")
                            continue
                        
                        for portfolio_key in portfolio_keys:
                            signal_file = os.path.join(portfolio_path, f"{portfolio_key}.csv")
                            if not os.path.exists(signal_file):
                                print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                                continue
                            
                            df = pd.read_csv(signal_file)
                            
                            # 거래 횟수
                            no_trade = len(df[df['Margin_Profit'] != 0])
                            
                            # 보유 기간 계산
                            df['Holding_Period'] = (df['Holding_Tickers'] != '').astype(int).cumsum()
                            max_holding_period = df[df['Holding_Tickers'] != '']['Holding_Period'].max() if no_trade > 0 else 0
                            max_holding_period = 0 if pd.isna(max_holding_period) else max_holding_period
                            mean_holding_period = df[df['Holding_Tickers'] != '']['Holding_Period'].mean() if no_trade > 0 else 0
                            mean_holding_period = 0 if pd.isna(mean_holding_period) else mean_holding_period
                            
                            # 승률
                            winning_ratio = len(df[df['Margin_Profit'] > 0]) / no_trade if no_trade > 0 else 0
                            
                            # 평균 수익/손실
                            profit_average = df[df['Margin_Profit'] > 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] > 0]) > 0 else 0
                            profit_average = 0 if pd.isna(profit_average) else profit_average
                            loss_average = df[df['Margin_Profit'] < 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] < 0]) > 0 else 0
                            loss_average = 0 if pd.isna(loss_average) else loss_average
                            
                            # 페이오프 비율
                            payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                            
                            # 수익률 요인
                            loss_sum = df[df['Margin_Profit'] < 0]['Margin_Profit'].sum()
                            profit_sum = df[df['Margin_Profit'] > 0]['Margin_Profit'].sum()
                            profit_factor = -profit_sum / loss_sum if loss_sum < 0 else 0
                            
                            # 최종 누적 수익 및 수익률
                            final_cumulative_profit = df['Cumulative_Profit'].iloc[-1]
                            final_cumulative_return = df['Cumulative_Return'].iloc[-1]
                            
                            # 최대 실현 수익/수익률
                            max_realized_profit = df['Margin_Profit'].max() if no_trade > 0 else 0
                            max_realized_profit = 0 if pd.isna(max_realized_profit) else max_realized_profit
                            max_realized_return = df['Margin_Return'].max() if no_trade > 0 else 0
                            max_realized_return = 0 if pd.isna(max_realized_return) else max_realized_return
                            
                            # 최대 손실 (MDD)
                            MDD = df['Drawdown'].min() if no_trade > 0 else 0
                            MDD = 0 if pd.isna(MDD) else MDD
                            MDD_rate = df['Drawdown_rate'].min() if no_trade > 0 else 0
                            MDD_rate = 0 if pd.isna(MDD_rate) else MDD_rate
                            
                            summary_data.append([
                                model, portfolio_type, portfolio_key, time_series_length, scenarios,
                                no_trade, max_holding_period, mean_holding_period, winning_ratio,
                                profit_average, loss_average, payoff_ratio, profit_factor,
                                final_cumulative_profit, final_cumulative_return,
                                max_realized_profit, max_realized_return,
                                MDD, MDD_rate
                            ])
        
        # 요약 데이터프레임 생성
        summary_df = pd.DataFrame(summary_data, columns=[
            "model", "portfolio_type", "portfolio_key", "train_test_length", "scenarios",
            "no_trade", "max_holding_period", "mean_holding_period", "winning_ratio",
            "profit_average", "loss_average", "payoff_ratio", "profit_factor",
            "final_cumulative_profit", "final_cumulative_return",
            "max_realized_profit", "max_realized_return",
            "MaxDrawdown", "MaxDrawdown_rate"
        ])
        
        summary_df = summary_df.round(3)
        
        # 출력 저장
        output_dir = os.path.join(base_output_path, "full_period")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "results_summary.csv")
        
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            key_columns = ["model", "portfolio_type", "portfolio_key", "train_test_length", "scenarios"]
            existing_keys = existing_df[key_columns].apply(tuple, axis=1)
            new_keys = summary_df[key_columns].apply(tuple, axis=1)
            non_duplicate = ~new_keys.isin(existing_keys)
            summary_df = summary_df[non_duplicate]
            if not summary_df.empty:
                summary_df.to_csv(output_file, mode='a', header=False, encoding='utf-8-sig', index=False)
                print(f"[INFO] Appended {len(summary_df)} new rows to {output_file}")
            else:
                print(f"[INFO] No new data to append to {output_file}")
        else:
            summary_df.to_csv(output_file, encoding='utf-8-sig', index=False)
            print(f"[INFO] Created and saved portfolio summary to {output_file}")
            
    def PlotResults(self):
        plt.rcParams.update({
            'axes.titlesize': 40,
            'axes.labelsize': 30,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'legend.fontsize': 20
        })

        base_input_path = "./stock_prediction/portfolio_strategy/portfolio_results/GRU_13_64_2/"
        base_output_path = "./stock_prediction/portfolio_strategy/plot/"
        model_name = "GRU_13_64_2"
        
        # 실험 파라미터
        TRAIN_MONTHS_LIST = [12, 24, 36]
        NUM_SCENARIOS_LIST = [25, 50, 75, 100]
        
        # 모델별 고정 색상 매핑
        model_colors = {
            'baseline':     '#7f7f7f' ,  # 검정
            'standard_gbm': '#f49ac2',  # 부드러운 보라 (pastel violet)
            'empirical_gbm':'#ffb347',  # 부드러운 주황 (pastel orange)
            'VanillaGAN':   '#fdfd96',  # 부드러운 노랑 (pastel yellow)
            'WGAN':         '#77dd77',  # 부드러운 초록 (pastel green)
            'QuantGAN':     '#aec6cf',  # 부드러운 파랑 (pastel blue)
            'DDPM':         '#c39bd3',  # 부드러운 남색/인디고 (pastel indigo)
            'LDM':          '#ff6961',  # 부드러운 빨강 (pastel red)
            'default':      '#000000'   # 회색 (fallback)
            }
                
        # 모든 조합식 순회
        for train_months in tqdm(TRAIN_MONTHS_LIST, desc="Processing train months"):
            for num_scenarios in NUM_SCENARIOS_LIST:
                time_series_length = f"Train_{train_months}_Test_{self.args.sliding_test_months}"
                
                model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
                
                # 포트폴리오 유형별 비교 데이터 수집
                comparison_data = {}
                
                for portfolio_type in ['all_tickers', 'sector', 'risk_category', 'market_cap']:
                    comparison_data[portfolio_type] = {}
                    portfolio_path = None
                    
                    for model in model_types:
                        method_input_path = os.path.join(base_input_path, model)
                        if model == "baseline":
                            input_base_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path)
                            output_dir_base = os.path.join(base_output_path, "{plot_type}", model, time_series_length)
                            scenarios = 0
                        else:
                            input_base_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{num_scenarios}", self.args.exp_root_path)
                            output_dir_base = os.path.join(base_output_path, "{plot_type}", model, time_series_length, f"Scenarios_{num_scenarios}")
                            scenarios = num_scenarios
                        
                        if not os.path.exists(input_base_path):
                            print(f"[WARNING] Input path {input_base_path} does not exist. Skipping.")
                            continue
                        
                        portfolio_path = os.path.join(input_base_path, portfolio_type)
                        if not os.path.exists(portfolio_path):
                            print(f"[WARNING] Portfolio path {portfolio_path} does not exist. Skipping.")
                            continue
                        
                        portfolio_keys = [f[:-4] for f in os.listdir(portfolio_path) if f.endswith('.csv')]
                        if not portfolio_keys:
                            print(f"[WARNING] No portfolio files found in {portfolio_path}. Skipping.")
                            continue
                        
                        for portfolio_key in portfolio_keys:
                            signal_file = os.path.join(portfolio_path, f"{portfolio_key}.csv")
                            if not os.path.exists(signal_file):
                                print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                                continue
                            
                            df = pd.read_csv(signal_file)
                            df['index'] = pd.to_datetime(df['index'])
                            df.set_index('index', inplace=True)
                            
                            # 디버깅: Holding_Tickers 결측값 확인
                            if df['Holding_Tickers'].isna().any():
                                print(f"[WARNING] NaN values found in Holding_Tickers for {portfolio_type}/{portfolio_key}, model={model}, Train_{train_months}_Scenarios_{num_scenarios}")
                                df['Holding_Tickers'] = df['Holding_Tickers'].fillna("")
                            
                            # 비교 플롯용 데이터 저장
                            comparison_data[portfolio_type][(model, portfolio_key)] = {
                                'Cumulative_Return': df['Cumulative_Return'],
                                'Drawdown_rate': df['Drawdown_rate']
                            }
                            
                            # 개별 플롯 생성
                            # 1. Cumulative Return Plot
                            ''' plt.figure(figsize=(30, 15))
                            plt.plot(df.index, df['Cumulative_Return'], label='Cumulative return', color='purple', linewidth=3.0)
                            plt.xlabel('Date')
                            plt.ylabel('Cumulative Return (%)')
                            plt.legend()
                            plt.grid(True)
                            
                            output_dir = output_dir_base.format(plot_type="Cumulative_plot")
                            os.makedirs(output_dir, exist_ok=True)
                            output_file = os.path.join(output_dir, f"{portfolio_key}.png")
                            plt.tight_layout()
                            plt.savefig(output_file)
                            plt.close()
                            print(f"[INFO] Saved cumulative return plot to {output_file}")
                            
                            # 2. Drawdown Plot
                            plt.figure(figsize=(30, 15))
                            plt.plot(df.index, df['Drawdown_rate'], label='Drawdown Rate', color='darkblue', linewidth=3.0)
                            plt.fill_between(df.index, 0, df['Drawdown_rate'], color='darkblue', alpha=0.3)
                            plt.xlabel('Date')
                            plt.ylabel('Drawdown Rate (%)')
                            plt.legend()
                            plt.grid(True)
                            
                            output_dir = output_dir_base.format(plot_type="Drawdown")
                            os.makedirs(output_dir, exist_ok=True)
                            output_file = os.path.join(output_dir, f"{portfolio_key}.png")
                            plt.tight_layout()
                            plt.savefig(output_file)
                            plt.close()
                            print(f"[INFO] Saved drawdown plot to {output_file}") '''
                            
                            # 3. Return Size Plot
                            plt.figure(figsize=(30, 15))
                            plt.axhline(y=0, color='gray', linestyle='--')
                            marker_size = 1000 * abs(df['Margin_Return'])
                            colors = ['red' if x >= 0 else 'blue' for x in df['Margin_Return']]
                            plt.scatter(df.index, df['Margin_Return'], s=marker_size, alpha=0.5, c=colors, label='Sell Signal Return')
                            plt.xlabel('Date')
                            plt.ylabel('Margin Return (%)')
                            plt.legend()
                            plt.grid(True)
                            
                            output_dir = output_dir_base.format(plot_type="Return_size")
                            os.makedirs(output_dir, exist_ok=True)
                            output_file = os.path.join(output_dir, f"{portfolio_key}.png")
                            plt.tight_layout()
                            plt.savefig(output_file)
                            plt.close()
                            print(f"[INFO] Saved return size plot to {output_file}")
                
                ''' # 포트폴리오 유형별 모델 비교 플롯
                for portfolio_type in comparison_data:
                    for portfolio_key in set(k[1] for k in comparison_data[portfolio_type].keys()):
                        # 1. Cumulative Return Comparison Plot
                        plt.figure(figsize=(30, 15))
                        
                        # baseline 모델 확인 및 포함
                        baseline_included = False
                        for (model, key), data in comparison_data[portfolio_type].items():
                            if key == portfolio_key:
                                if model == 'baseline':
                                    baseline_included = True
                                # 모델별 고정 색상 및 라인 굵기 사용
                                color = model_colors.get(model, model_colors['default'])
                                plt.plot(data['Cumulative_Return'].index, data['Cumulative_Return'], 
                                        label=model, color=color, linewidth=3.0)
                        
                        # baseline 데이터가 없으면 경고
                        if not baseline_included:
                            print(f"[WARNING] Baseline model not found for {portfolio_type}/{portfolio_key}, Train_{train_months}_Scenarios_{num_scenarios}")
                        
                        plt.xlabel('Date')
                        plt.ylabel('Cumulative Return (%)')
                        plt.title(f'Cumulative Return Comparison')
                        plt.legend()
                        plt.grid(True)
                        
                        output_dir = os.path.join(base_output_path, "Cumulative_comparison", f"Train_{train_months}_Scenarios_{num_scenarios}", portfolio_type)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"{portfolio_key.replace(' ', '_')}.png")
                        plt.tight_layout()
                        plt.savefig(output_file)
                        plt.close()
                        print(f"[INFO] Saved cumulative return comparison plot to {output_file}")
                        
                        # 2. Drawdown Rate Comparison Plot
                        plt.figure(figsize=(30, 15))
                        
                        # LDM을 먼저 플로팅하기 위해 모델 순서 재정렬
                        baseline_included = False
                        # LDM 먼저 처리
                        if ('LDM', portfolio_key) in comparison_data[portfolio_type]:
                            model = 'LDM'
                            data = comparison_data[portfolio_type][(model, portfolio_key)]
                            color = model_colors.get(model, model_colors['default'])
                            plt.plot(data['Drawdown_rate'].index, data['Drawdown_rate'], 
                                    label=model, color=color, linewidth=3.0)
                            plt.fill_between(data['Drawdown_rate'].index, 0, data['Drawdown_rate'], 
                                            color=color, alpha=0.3)
                        
                        # 나머지 모델들 플로팅
                        for (model, key), data in comparison_data[portfolio_type].items():
                            if key == portfolio_key and model != 'LDM':
                                if model == 'baseline':
                                    baseline_included = True
                                color = model_colors.get(model, model_colors['default'])
                                plt.plot(data['Drawdown_rate'].index, data['Drawdown_rate'], 
                                        label=model, color=color, linewidth=3.0)
                                plt.fill_between(data['Drawdown_rate'].index, 0, data['Drawdown_rate'], 
                                                color=color, alpha=0.3)
                        
                        # baseline 데이터가 없으면 경고
                        if not baseline_included:
                            print(f"[WARNING] Baseline model not found for {portfolio_type}/{portfolio_key}, Train_{train_months}_Scenarios_{num_scenarios}")
                        
                        plt.xlabel('Date')
                        plt.ylabel('Drawdown Rate (%)')
                        plt.title(f'Drawdown Rate Comparison')
                        plt.legend()
                        plt.grid(True)
                        
                        output_dir = os.path.join(base_output_path, "Drawdown_comparison", f"Train_{train_months}_Scenarios_{num_scenarios}", portfolio_type)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"{portfolio_key.replace(' ', '_')}.png")
                        plt.tight_layout()
                        plt.savefig(output_file)
                        plt.close()
                        print(f"[INFO] Saved drawdown rate comparison plot to {output_file}") '''