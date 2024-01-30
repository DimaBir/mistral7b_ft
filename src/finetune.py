import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer
from model import load_model
from dataset import CodingDataset
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def finetune(cfg: DictConfig):
    """
    Fine-tune the Mistral 7B model with LoRA and TensorBoard integration.

    Parameters:
    cfg (DictConfig): The configuration object from Hydra.
    """
    print(f"HERE!!!!!")
    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Load model and dataset
    model = load_model(cfg)  # This now returns a PEFTModel with LoRA
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    train_dataset = CodingDataset(tokenizer, 'train', cfg.model.max_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.train_batch_size)

    optimizer = AdamW(model.parameters(), lr=cfg.model.learning_rate)

    for epoch in range(cfg.model.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            # Ensure the input is correctly formatted for the PEFTModel
            input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate and log average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('training_loss', avg_loss, epoch)

        print(f'Epoch {epoch+1}/{cfg.model.epochs} completed. Avg Loss: {avg_loss}')

    # Save the underlying base model of PEFTModel
    torch.save(model.base_model.state_dict(), cfg.paths.model_save_path)
    writer.close()
