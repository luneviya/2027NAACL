import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def model_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table


class Model(BaseModel):
    """
    ReFCode bi-encoder with uncertainty heads and a code-text relevance head.

    Default forward() mean-pools the last hidden layer and L2-normalizes it,
    so retrieval evaluation remains a standard bi-encoder cosine pipeline.

    encode_inputs(..., return_uncertainty=True) returns:
      - z: deterministic normalized representation from the last layer mean-pool
      - z_plus: uncertainty sampled representation selected by max cosine(z, sample)
      - mu/logvar: Gaussian parameters for KL regularization
    """

    def __init__(self, encoder, hidden_size=None, dropout=0.1):
        super(Model, self).__init__()
        self.encoder = encoder
        if hidden_size is None:
            hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", 768)
        self.hidden_size = hidden_size
        self.uncertainty_dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden_size * 4, hidden_size)
        self.logvar_head = nn.Linear(hidden_size * 4, hidden_size)

        # CTRD = Code-Text Relevance Detection.
        # Pair feature follows the common fine-grained matching form:
        # [q, c, |q-c|, q*c] -> binary related / unrelated.
        self.ctrd_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self._init_uncertainty_heads()
        self._init_ctrd_head()

    def _init_uncertainty_heads(self):
        # Small init: start close to the deterministic encoder space.
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.5)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=0.1)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def _init_ctrd_head(self):
        # Conservative initialization for the classifier head itself, while the
        # default script uses an aggressive loss weight to make CTRD matter.
        for module in self.ctrd_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def ctrd_logits(self, query_vec, code_vec):
        """Return relevance logits for flattened query-code vector pairs.

        query_vec and code_vec must have the same first dimension. They are
        expected to be L2-normalized retrieval representations. This method is
        intentionally separate from forward(), so retrieval evaluation remains
        exactly the original bi-encoder cosine pipeline.
        """
        pair_vec = torch.cat([
            query_vec,
            code_vec,
            torch.abs(query_vec - code_vec),
            query_vec * code_vec,
        ], dim=-1)
        return self.ctrd_head(pair_vec).squeeze(-1)

    @staticmethod
    def _mean_pool(hidden, input_ids):
        mask = input_ids.ne(1).float()  # UniXcoder pad id is 1.
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        return (hidden * mask[:, :, None]).sum(1) / denom

    def _encoder_forward(self, input_ids):
        attention_mask = input_ids.ne(1)
        try:
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
        except TypeError:
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs[0]
            hidden_states = outputs[-1]
        return last_hidden, hidden_states

    def encode_inputs(self, input_ids, return_uncertainty=False, num_samples=4):
        last_hidden, hidden_states = self._encoder_forward(input_ids)
        last_avg = self._mean_pool(last_hidden, input_ids)
        z = F.normalize(last_avg, p=2, dim=1)

        if not return_uncertainty:
            return z

        # ReFCode pooling design: concatenate multiple views of token-level states.
        first_avg = self._mean_pool(hidden_states[0], input_ids)
        last_two_avg = 0.5 * (
            self._mean_pool(hidden_states[-1], input_ids) +
            self._mean_pool(hidden_states[-2], input_ids)
        )
        first_last_avg = 0.5 * (first_avg + last_avg)
        cls_last = last_hidden[:, 0, :]
        z_prime = torch.cat([last_avg, first_last_avg, last_two_avg, cls_last], dim=-1)
        z_prime = self.uncertainty_dropout(z_prime)

        mu = self.mu_head(z_prime)
        logvar = self.logvar_head(z_prime).clamp(min=-10.0, max=4.0)
        std = torch.exp(0.5 * logvar)

        num_samples = max(1, int(num_samples))
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), device=mu.device, dtype=mu.dtype)
        samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        samples = F.normalize(samples, p=2, dim=-1)
        sims = (samples * z.unsqueeze(0)).sum(dim=-1)  # [S, B]
        best = sims.argmax(dim=0)
        batch_idx = torch.arange(mu.size(0), device=mu.device)
        z_plus = samples[best, batch_idx]

        return {
            "z": z,
            "z_plus": z_plus,
            "mu": mu,
            "logvar": logvar,
        }

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            return self.encode_inputs(code_inputs, return_uncertainty=False)
        return self.encode_inputs(nl_inputs, return_uncertainty=False)
