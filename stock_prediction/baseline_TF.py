from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.feature_preprocessing import calculate_indicators, add_labels
import math

plt.rcParams.update({
    'axes.titlesize': 25,    # 제목 폰트 크기
    'axes.labelsize': 20,    # Y축 레이블 폰트 크기
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

        # 유효한 인덱스 생성
        self.valid_indices = []
        for i in range(len(self.data_df) - self.seq_len):
            self.valid_indices.append(('data', i))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        _, start_idx = self.valid_indices[idx]
        seq_data = self.data_df[self.feature_cols].iloc[start_idx:start_idx + self.seq_len].values
        label = self.data_df[self.label_col].iloc[start_idx + self.seq_len]

        seq_data = torch.tensor(seq_data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return seq_data, label

# Positional Encoding 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer 모델
class StockTransformer(nn.Module):
    def __init__(self, input_dim=13, d_model=64, nhead=4, num_layers=3, dropout=0.1, seq_len=127):
        super(StockTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        x = self.input_linear(x)
        residual = x  # 잔차 연결을 위한 입력 저장
        x = self.pos_encoder(x)
        
        if return_attention:
            attn_weights = []
            for layer in self.transformer_encoder.layers:
                x, attn = layer.self_attn(x, x, x, need_weights=True)
                attn_weights.append(attn)
            attn_weights = attn_weights[-1]  # [batch, seq_len, seq_len]
            x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x)

        x = x + residual  # 잔차 연결
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        if return_attention:
            return x, attn_weights
        return x

    def get_model_name(self):
        return f"Transformer_{self.input_dim}_{self.d_model}_{self.num_layers}"

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    scaler = GradScaler()
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
            
            with autocast():
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
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
            
class Exp_STP_Baseline_TF:
    def __init__(self, args):
        self.args = args
    
    def prepare_train_data(self, train_df, seq_len):
        train_df = train_df.copy()
        train_df['index'] = pd.to_datetime(train_df['index'])
        
        train_df = calculate_indicators(train_df)
        train_df = add_labels(train_df)
        
        indicator_columns = [
            'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist', 
            'MOM', 'PPO', 'ROC', 'ROCR', 'RSI', 
            'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'Up_Down'
        ]
        processed_train_df = train_df[indicator_columns]
        processed_train_df = processed_train_df.dropna().reset_index(drop=True)
        
        return processed_train_df

    def prepare_test_data(self, train_df, test_df, seq_len):
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
        
        indicator_columns = [
            'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist', 
            'MOM', 'PPO', 'ROC', 'ROCR', 'RSI', 
            'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'Up_Down'
        ]
        processed_test_df = processed_test_df[indicator_columns]
        processed_test_df = processed_test_df.dropna().reset_index(drop=True)
        
        return processed_test_df

    def process(self):
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]
            print(f"\n[INFO] Simulating for sector: {self.args.sector} | ticker: {ticker_name}")
            
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
                processed_train_df = self.prepare_train_data(
                    train_df,
                    seq_len=self.args.seq_len
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
                
                model = StockTransformer(input_dim=len(train_dataset.feature_cols), seq_len=self.args.seq_len).to(device)
                model_name = model.get_model_name()
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

                print(f"[INFO] Training model for {self.args.sector} | {ticker_name} | Window {window_idx+1}")
                train_model(model, train_loader, criterion, optimizer, self.args.num_epochs, device)

                print(f"[INFO] Testing model for {self.args.sector} | {ticker_name} | Window {window_idx+1}")
                results = test_model(model, test_loader, criterion, device)

                # 결과 저장
                results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
                save_path = f'./stock_prediction/pred_results/{model_name}/origin_data/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/'
                os.makedirs(save_path, exist_ok=True)
                results_df.to_csv(f'{save_path}window_{window_idx+1}.csv', index=False)