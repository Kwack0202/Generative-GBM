from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.feature_preprocessing import calculate_indicators, add_labels

plt.rcParams.update({
    'axes.titlesize': 25,    # 제목 폰트 크기
    'axes.labelsize': 20,    # Y축 레이블 폰트 크기 (metric 강조)
    'xtick.labelsize': 15,   # X축 눈금 레이블 폰트 크기
    'ytick.labelsize': 15,   # Y축 눈금 레이블 폰트 크기
    'legend.fontsize': 20    # 범례 폰트 크기
})

# 데이터셋 클래스
class StockScenarioDataset(Dataset):
    def __init__(self, data_df, seq_len=127, mode='train'):
        self.data_df = data_df
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [
            'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist',
            'MOM', 'PPO', 'ROC', 'ROCR', 'RSI',
            'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX'
        ]
        self.label_col = 'Up_Down'

        # 데이터 전처리: 결측값 처리 및 정규화
        self.data_df[self.feature_cols] = self.data_df[self.feature_cols].fillna(0)
        self.data_df[self.feature_cols] = self.data_df[self.feature_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.data_df[self.feature_cols] = scaler.fit_transform(self.data_df[self.feature_cols])

        # 시나리오별로 데이터 분리 (train 모드일 때만)
        if mode == 'train':
            self.scenario_groups = self.data_df.groupby('scenario_id')
            self.scenario_ids = list(self.scenario_groups.groups.keys())
        else:
            self.scenario_groups = None
            self.scenario_ids = ['test']

        # 유효한 인덱스 생성
        self.valid_indices = []
        if mode == 'train':
            for scenario_id in self.scenario_ids:
                scenario_data = self.scenario_groups.get_group(scenario_id)
                for i in range(len(scenario_data) - self.seq_len):
                    self.valid_indices.append((scenario_id, i))
        else:
            for i in range(len(self.data_df) - self.seq_len):
                self.valid_indices.append(('test', i))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        scenario_id, start_idx = self.valid_indices[idx]
        if self.mode == 'train':
            scenario_data = self.scenario_groups.get_group(scenario_id)
        else:
            scenario_data = self.data_df

        # seq_len 길이의 시퀀스 추출
        seq_data = scenario_data[self.feature_cols].iloc[start_idx:start_idx + self.seq_len].values
        label = scenario_data[self.label_col].iloc[start_idx + self.seq_len]

        seq_data = torch.tensor(seq_data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return seq_data, label

class StockGRU(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, dropout=0.2):
        super(StockGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU 레이어
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        gru_out, _ = self.gru(x)  # gru_out: [batch, seq_len, hidden_dim]
        x = gru_out[:, -1, :]  # 마지막 타임스텝 출력 사용
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 1]
        return x

    def get_model_name(self):
        return f"GRU_{self.input_dim}_{self.hidden_dim}_{self.num_layers}"
    
# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    scaler = GradScaler()  # Mixed Precision 스케일러
    progress = tqdm(range(num_epochs))
    
    for epoch in progress:
        running_loss = 0.0
        all_labels = []
        all_preds = []
        total = 0
        correct = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():  # Mixed Precision 컨텍스트
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()  # 스케일링된 그래디언트
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        msg = f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}"
        progress.set_description(msg)
    
    torch.cuda.empty_cache()

# 테스트 함수
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            results.extend(zip(labels.cpu().numpy(), probabilities.cpu().numpy()))

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    torch.cuda.empty_cache()
    return results

# 모델 저장 함수
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
            
class Exp_STP_GRU:
    def __init__(self, args):
        self.args = args
    
    def select_balanced_scenarios(self, scenario_df, num_scenarios):
        scenario_columns = scenario_df.columns

        # 1. 로그 수익률 계산 및 드리프트/변동성 추출
        scenario_stats = []
        for scenario_id in scenario_columns:
            prices = scenario_df[scenario_id].values
            log_returns = np.diff(np.log(prices))  # 로그 수익률
            drift = np.mean(log_returns)  # 평균 (드리프트)
            volatility = np.std(log_returns)  # 표준편차 (변동성)
            scenario_stats.append({'scenario_id': scenario_id, 'Drift': drift, 'Volatility': volatility})
        
        stats_df = pd.DataFrame(scenario_stats)
        
        # 2. 드리프트와 변동성의 분위수 계산
        drift_quantiles = stats_df['Drift'].quantile([0.333, 0.667]).values
        volatility_quantiles = stats_df['Volatility'].quantile([0.333, 0.667]).values
        
        # 3. 위험군 분류 함수
        def classify_risk(drift, volatility):
            if drift <= drift_quantiles[0]:
                drift_level = 'Low'
            elif drift <= drift_quantiles[1]:
                drift_level = 'Medium'
            else:
                drift_level = 'High'
            
            if volatility <= volatility_quantiles[0]:
                vol_level = 'Low'
            elif volatility <= volatility_quantiles[1]:
                vol_level = 'Medium'
            else:
                vol_level = 'High'
            
            if vol_level == 'High' and drift_level == 'Low':
                return 'High Risk'
            elif vol_level == 'Low' and drift_level in ['Medium', 'High']:
                return 'Low Risk'
            else:
                return 'Medium Risk'
        
        # 4. 각 시나리오에 위험도 할당
        stats_df['Risk'] = stats_df.apply(lambda row: classify_risk(row['Drift'], row['Volatility']), axis=1)
        
        # 5. 위험도별 시나리오 비율에 따른 선택 (25% Low, 50% Medium, 25% High)
        risk_groups = stats_df.groupby('Risk')
        selected_scenarios = []
        
        # 각 위험도 그룹별 목표 시나리오 수 계산
        low_risk_target = int(num_scenarios * 0.25)  # 25% for Low Risk
        medium_risk_target = int(num_scenarios * 0.50)  # 50% for Medium Risk
        high_risk_target = int(num_scenarios * 0.25)  # 25% for High Risk
        
        # 각 위험도 그룹에서 시나리오 선택
        for risk, target_count in [
            ('Low Risk', low_risk_target),
            ('Medium Risk', medium_risk_target),
            ('High Risk', high_risk_target)
        ]:
            if risk in risk_groups.groups:
                group = risk_groups.get_group(risk)
                # 그룹 내 시나리오 수가 목표보다 적을 경우, 가능한 모든 시나리오 선택
                sample_count = min(len(group), target_count)
                if sample_count > 0:
                    selected = group.sample(n=sample_count, replace=False, random_state=42)
                    selected_scenarios.extend(selected['scenario_id'].tolist())
            else:
                print(f"[WARNING] {risk} 그룹에 시나리오가 없습니다.")
        
        # 6. 부족한 시나리오를 나머지 시나리오에서 랜덤하게 선택
        if len(selected_scenarios) < num_scenarios:
            shortfall = num_scenarios - len(selected_scenarios)
            remaining_scenarios = stats_df[~stats_df['scenario_id'].isin(selected_scenarios)]
            if len(remaining_scenarios) > 0:
                additional = remaining_scenarios.sample(
                    n=min(shortfall, len(remaining_scenarios)), 
                    replace=False, 
                    random_state=42
                )
                selected_scenarios.extend(additional['scenario_id'].tolist())
            else:
                print(f"[WARNING] 추가로 선택할 수 있는 시나리오가 없습니다. (현재 선택된 시나리오 수: {len(selected_scenarios)}/{num_scenarios})")
        
        return selected_scenarios, stats_df, drift_quantiles, volatility_quantiles
    
    def prepare_scenario_data(self, train_df, scenario_df, seq_len, num_scenarios):
        train_df = train_df.copy()
        train_df['index'] = pd.to_datetime(train_df['index'])
        
        scenario_df = scenario_df.copy()
        
        all_scenarios = []
        selected_scenarios, _, _, _ = self.select_balanced_scenarios(scenario_df, num_scenarios)
        
        for scenario_id in selected_scenarios:
            temp_df = train_df.copy()
            
            scenario_series = pd.Series(scenario_df[scenario_id].values, name='Adj Close')
            scenario_temp_df = pd.DataFrame({
                'index': pd.date_range(start=train_df['index'].iloc[-1] + pd.Timedelta(days=1),
                                    periods=len(scenario_series), freq='D'),
                'Adj Close': scenario_series
            })
            
            combined_df = pd.concat([temp_df, scenario_temp_df], ignore_index=True)
            combined_df = combined_df.sort_values('index').reset_index(drop=True)
            
            combined_df = calculate_indicators(combined_df)
            combined_df = add_labels(combined_df)
            
            # 기술적 지표와 라벨만 선택
            indicator_columns = [
                'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist', 
                'MOM', 'PPO', 'ROC', 'ROCR', 'RSI', 
                'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'Up_Down'
            ]

            scenario_start_idx = len(train_df)
            processed_scenario_df = combined_df.iloc[scenario_start_idx - seq_len:].reset_index(drop=True)
            processed_scenario_df = processed_scenario_df[indicator_columns]
            
            processed_scenario_df = processed_scenario_df.dropna().reset_index(drop=True)
            processed_scenario_df['scenario_id'] = scenario_id
            
            all_scenarios.append(processed_scenario_df)
        
        processed_train_df = pd.concat(all_scenarios, ignore_index=True)
        
        return processed_train_df

    def prepare_test_data(self, train_df, test_df, seq_len):
        # 'index'를 datetime으로 변환
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['index'] = pd.to_datetime(train_df['index'])
        test_df['index'] = pd.to_datetime(test_df['index'])
        
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df = combined_df.sort_values('index').reset_index(drop=True)
        
        combined_df = calculate_indicators(combined_df)
        combined_df = add_labels(combined_df)
        
        test_start_idx = len(train_df)
        
        processed_test_df = combined_df.iloc[test_start_idx - seq_len:].reset_index(drop=True)
        
        # 기술적 지표와 라벨만 선택
        indicator_columns = [
            'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist', 
            'MOM', 'PPO', 'ROC', 'ROCR', 'RSI', 
            'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'Up_Down'
        ]
        processed_test_df = processed_test_df[indicator_columns]
        processed_test_df = processed_test_df.dropna().reset_index(drop=True)
        
        return processed_test_df

    def process(self):
        base_path = f'./outputs/{self.args.GBM_type}'
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]
            print(f"\n[INFO] Simulating for sector : {self.args.sector} | ticker: {ticker_name}")
            
            stock_data = pd.read_csv(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector, stock_file))
            
            sliding_windows = get_sliding_window_data(
                stock_data,
                date_col='index',
                test_start_year=self.args.test_start_year,
                total_test_months=self.args.total_test_months,
                sliding_test_months=self.args.sliding_test_months,
                train_months=self.args.train_months
            )
            
            for window_idx, (train_df, test_df, window_info) in enumerate(sliding_windows):
                # Load scenario data
                scenario_df = pd.read_csv(os.path.join(
                    base_path,
                    f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/',
                    f'GBM_scenario/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/',
                    f"Sliding_Window_{window_idx+1}_simulated_paths.csv"
                ))
                
                # Get scenario stats and quantiles for visualization
                selected_scenarios, stats_df, drift_quantiles, volatility_quantiles = self.select_balanced_scenarios(
                    scenario_df, self.args.num_scenarios
                )
                
                # Visualization
                plt.figure(figsize=(20, 15))
                colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
                
                # Plot all scenarios with their respective risk category colors
                for category in stats_df['Risk'].unique():
                    subset = stats_df[stats_df['Risk'] == category]
                    plt.scatter(subset['Drift'], subset['Volatility'], 
                                c=colors[category], label=category, alpha=0.6, s=200)
                
                # Highlight selected scenarios with larger markers and black borders
                selected_subset = stats_df[stats_df['scenario_id'].isin(selected_scenarios)]
                for category in selected_subset['Risk'].unique():
                    subset = selected_subset[selected_subset['Risk'] == category]
                    plt.scatter(subset['Drift'], subset['Volatility'], 
                                c=colors[category], 
                                s=400,  # Larger size for emphasis
                                edgecolors='black',  # Black border
                                linewidths=2,  # Border thickness
                                alpha=0.8, 
                                label=f'Selected {category}' if category in stats_df['Risk'].unique() else None)
                
                # Add scenario ID labels for all points
                ''' for i, row in stats_df.iterrows():
                    plt.annotate(row['scenario_id'], (row['Drift'], row['Volatility']), 
                                fontsize=10, alpha=0.8, xytext=(5, 5), textcoords='offset points') '''
                
                # Add grid lines
                plt.axvline(x=drift_quantiles[0], color='gray', linestyle='--', alpha=0.5)
                plt.axvline(x=drift_quantiles[1], color='gray', linestyle='--', alpha=0.5)
                plt.axhline(y=volatility_quantiles[0], color='gray', linestyle='--', alpha=0.5)
                plt.axhline(y=volatility_quantiles[1], color='gray', linestyle='--', alpha=0.5)
                
                # Set axis labels and title
                plt.xlabel('Drift')
                plt.ylabel('Volatility')
                plt.title(f'Scenario Risk Classification for {self.args.sector} - {ticker_name} (Window {window_idx+1})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                save_path = f'./stock_prediction/risk_plots/{self.args.GBM_type}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/Scenarios_{self.args.num_scenarios}/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/'
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}window_{window_idx+1}_risk_classification.png', bbox_inches='tight')
                plt.close()  # Close the figure to free memory
                
                processed_train_df = self.prepare_scenario_data(
                    train_df,
                    scenario_df,
                    seq_len=self.args.seq_len,
                    num_scenarios=self.args.num_scenarios
                )
                
                processed_test_df = self.prepare_test_data(
                    train_df,
                    test_df,
                    seq_len=self.args.seq_len  
                )
                
                train_dataset = StockScenarioDataset(processed_train_df, seq_len=self.args.seq_len, mode='train')
                test_dataset = StockScenarioDataset(processed_test_df, seq_len=self.args.seq_len, mode='test')
                train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
                                        
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                model = StockGRU(input_dim=len(train_dataset.feature_cols)).to(device)
                model_name = model.get_model_name()
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

                print(f"[INFO] Training model for Scenarios {self.args.num_scenarios} | {self.args.sector} | {ticker_name} | Window {window_idx+1}")
                train_model(model, train_loader, criterion, optimizer, self.args.num_epochs, device)

                # 모델 저장 경로에 num_scenarios 반영
                # save_path = f'./stock_prediction/saved_model/{model_name}/{self.args.GBM_type}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/Scenarios_{self.args.num_scenarios}/{self.args.exp_root_path}/{ticker_name}/window_{window_idx+1}.pth'
                # save_model(model, save_path)

                print(f"[INFO] Testing model for Scenarios {self.args.num_scenarios} | {self.args.sector} | {ticker_name} | Window {window_idx+1}")
                results = test_model(model, test_loader, criterion, device)

                # 결과 저장 경로에 num_scenarios 반영
                results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
                os.makedirs(f'./stock_prediction/pred_results/{model_name}/{self.args.GBM_type}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/Scenarios_{self.args.num_scenarios}/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/', exist_ok=True)
                results_df.to_csv(f'./stock_prediction/pred_results/{model_name}/{self.args.GBM_type}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/Scenarios_{self.args.num_scenarios}/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/window_{window_idx+1}.csv', index=False)
    