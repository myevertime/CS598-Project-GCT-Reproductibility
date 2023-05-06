class Args:
    def __init__(self, prediction_task: str):
        if prediction_task == "expired":
            self.label_key = "expired"
            self.learning_rate = 0.00011
            self.reg_coef = 1.5
            self.hidden_dropout = 0.72
        elif prediction_task == "readmission":
            self.label_key = "readmission"
            self.learning_rate = 0.00022
            self.reg_coef = 0.1
            self.hidden_dropout = 0.08
        else:
            raise ValueError("Invalid prediction task: {}".format(prediction_task))

        # Training arguments
        self.max_steps = 1000000
        self.warmup = 0.05  # default
        self.logging_steps = 100  # default
        self.num_train_epochs = 1  # default
        self.seed = 42  # default

        # Model parameters arguments
        self.embedding_dim = 128
        self.max_num_codes = 50
        self.num_stacks = 3
        self.batch_size = 32
        self.prior_scalar = 0.5
        self.num_heads = 1

        # save and load the cache/dataset/env path (required)
        self.fold = 0
        self.data_dir = "eicu_data"
        self.eicu_csv_dir = "../eicu_csv"
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.output_dir = "eicu_output/model_pyhealth_" + timestamp
        self.output_dir = "eicu_output/model_pyhealth_" + self.label_key + "_2"

        # save and load the models (optional)
        self.save_model = True
        self.load_prev_model = False
        self.prev_model_path = "eicu_output/model_pyhealth_" + self.label_key + "/model.pt"


args = Args("readmission")
