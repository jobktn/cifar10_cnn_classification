
class ModelBuilder(object):
    def __init__(self, model, loss_fn, optimizer, print_loss_freq=10):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        # self.writer = None

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []  
        self.val_accuracies = []    
        self.total_epochs = 0

        # sets frequency of printing the losses
        self.print_loss_freq = print_loss_freq

        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        # Returns the function that will be called inside the train loop
        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()
        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            # print('Loss in minibatch = %.2f' %mini_batch_loss)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)


    def calculate_accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                images, labels = data
                outputs = self.model(images.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum().item()
        return correct / total

    def train(self, n_epochs, seed=42, target_accuracy=0.75):
        self.set_seed(seed)
        for epoch in range(n_epochs):
            # Perform training
            loss = self._mini_batch(validation=False)
            self.train_losses.append(loss)
            train_acc = self.calculate_accuracy(self.train_loader)
            self.train_accuracies.append(train_acc)  # Store training accuracy
        
            # Perform validation
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
                val_acc = self.calculate_accuracy(self.val_loader)
                self.val_accuracies.append(val_acc)  # Store validation accuracy
        
            # Print the progress every few epochs
            if (epoch + 1) % self.print_loss_freq == 0:
                print(f"Epoch {epoch+1}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc >= target_accuracy:
                print(f"Early stopping at epoch {epoch+1} as validation accuracy reached {val_acc*100:.2f}%")
                break

    def predict(self, x):
        # Set it to evaluation mode for predictions
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig,ax = plt.subplots(1,1,figsize=(10, 4))
        ax.plot(self.train_losses, label='Training Loss', c='b')
        ax.plot(self.val_losses, label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        fig.legend()
        fig.tight_layout()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'train_loss': self.train_losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.train_losses = checkpoint['train_loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training

    # modified plot accuracy fn
    def plot_accuracies(self):
        fig,ax = plt.subplots(1,1,figsize=(10, 4))
        ax.plot(self.train_accuracies, label='Training Accuracy', c='b')
        ax.plot(self.val_accuracies, label='Validation Accuracy', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        fig.legend()
        fig.tight_layout()

    # modified confusion matrix fn
    def generate_confusion_matrix(self, loader):
        data_preds = []
        data_labels = []

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Make predictions
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                # Collect predictions and labels
                data_preds.extend(predicted.cpu().numpy())
                data_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays for easier handling
        data_preds = np.array(data_preds)
        data_labels = np.array(data_labels)

        # Generate confusion matrix
        # cm = confusion_matrix(all_labels, all_preds)
        return data_preds, data_labels #cm
        
        
    def plot_confusion_matrix(self,  data_labels, data_preds, labels=[0, 1, 2, 3, 4]):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Raw confusion matrix
        ConfusionMatrixDisplay.from_predictions( data_labels, data_preds, labels=labels, ax=ax[0], colorbar=False)
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("True")

        # Normalized confusion matrix
        ConfusionMatrixDisplay.from_predictions(data_labels, data_preds, labels=labels, normalize='all', ax=ax[1], colorbar=False)
        ax[1].set_title("Confusion Matrix (Normalised)")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("True")

        plt.tight_layout()
        plt.show()

