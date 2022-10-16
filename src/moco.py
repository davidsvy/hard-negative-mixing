import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Based on:
    https://arxiv.org/abs/1911.05722
    https://proceedings.neurips.cc/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf
    """

    def __init__(
            self, encoder, dim=128, K=2 ** 12, m=0.999, T=0.07, n_hard=32,
            s1_hard=16, s2_hard=16, start1_hard=2 ** 10, start2_hard=2 ** 11,
    ):
        super(MoCo, self).__init__()

        self.K, self.m, self.T = K, m, T
        self.n_hard, self.s1_hard, self.s2_hard = n_hard, s1_hard, s2_hard
        self.start1_hard, self.start2_hard = start1_hard, start2_hard
        self.enable1_hard = s1_hard > 0 and start1_hard > 0
        self.enable2_hard = s2_hard > 0 and start2_hard > 0
        self.beta_hard = 0.5

        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        queue = F.normalize(torch.randn(K, dim), dim=1)
        self.register_buffer('queue', queue)
        self.queue_ptr = 0

        queue_file = torch.full(size=(1, K), fill_value=-1, dtype=torch.int32)
        self.register_buffer('queue_file', queue_file)

    def get_encoder(self):
        return copy.deepcopy(self.encoder_q)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, file_idxs):
        # keys -> [batch_size, d_out]
        batch_size = keys.shape[0]

        self.queue[self.queue_ptr: self.queue_ptr + batch_size, :] = keys
        self.queue_file[
            0, self.queue_ptr: self.queue_ptr + batch_size] = file_idxs

        self.queue_ptr = (self.queue_ptr + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # x -> [batch_size, ...]
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        x = x[idx_shuffle]

        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # x -> [batch_size, ...]
        # idx_unshuffle -> [batch_size]
        return x[idx_unshuffle]

    @torch.no_grad()
    def apply_encoder_k(self, k):
        k, idx_unshuffle = self._batch_shuffle(k)

        out_k = F.normalize(self.encoder_k(k), dim=1)
        out_k = self._batch_unshuffle(out_k, idx_unshuffle=idx_unshuffle)

        return out_k

    def find_hard_negatives(self, logits):
        # logits -> [batch_size, len_queue + 1]
        _, idxs_hard = torch.topk(
            logits.clone().detach(), k=self.n_hard, dim=-1, sorted=False)
        # idxs_hard -> [batch_size, n_hard]

        return idxs_hard

    def hard_negatives1(self, out_q, logits, idxs_hard):
        # out_q -> [batch_size, d_out]
        # logits -> [batch_size, len_queue + ...]
        # idxs_hard -> [batch_size, n_hard]
        batch_size, device = out_q.shape[0], out_q.device

        idxs1, idxs2 = torch.randint(
            0, self.n_hard, size=(2, batch_size, self.s1_hard), device=device)
        # idxs1, idxs2 -> [batch_size, s1_hard]
        alpha = torch.rand(size=(batch_size, self.s1_hard, 1), device=device)
        # alpha -> [batch_size, s1_hard, 1]

        neg1_hard = self.queue[
            torch.gather(idxs_hard, dim=1, index=idxs1)].clone().detach()
        neg2_hard = self.queue[
            torch.gather(idxs_hard, dim=1, index=idxs2)].clone().detach()
        # neg1_hard, neg2_hard -> [batch_size, s1_hard, d_out]

        neg_hard = alpha * neg1_hard + (1 - alpha) * neg2_hard
        neg_hard = F.normalize(neg_hard, dim=-1).detach()
        # neg_hard -> [batch_size, s1_hard, d_out]

        logits_hard = torch.einsum(
            'b d, b s d -> b s', out_q, neg_hard) / self.T
        # logits_hard -> [batch_size, s1_hard]

        logits = torch.cat([logits, logits_hard], dim=1)
        # logits -> [batch_size, len_queue + ... + s1_hard]

        return logits

    def hard_negatives2(self, out_q, logits, idxs_hard):
        # out_q -> [batch_size, d_out]
        # logits -> [batch_size, len_queue + ...]
        # idxs_hard -> [batch_size, n_hard]
        batch_size, device = out_q.shape[0], out_q.device

        idxs = torch.randint(
            0, self.n_hard, size=(batch_size, self.s2_hard), device=device)
        # idxs -> [batch_size, s2_hard]
        beta = torch.rand(
            size=(batch_size, self.s2_hard, 1), device=device) * self.beta_hard
        # beta -> [batch_size, s2_hard, 1]

        neg_hard = self.queue[
            torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        # neg_hard -> [batch_size, s2_hard, d_out]
        neg_hard = beta * \
            out_q.clone().detach()[:, None] + (1 - beta) * neg_hard
        neg_hard = F.normalize(neg_hard, dim=-1).detach()
        # neg_hard -> [batch_size, s2_hard, d_out]

        logits_hard = torch.einsum(
            'b d, b s d -> b s', out_q, neg_hard) / self.T
        # logits_hard -> [batch_size, s2_hard]

        logits = torch.cat(
            [logits, logits_hard], dim=1)
        # logits -> [batch_size, len_queue + ... + s2_hard]

        return logits

    def forward(self, q, k, step, file_idxs=None):
        # q, k -> [batch_size, n_channels, height, width]
        # file_idxs -> [batch_size]
        batch_size, device = q.shape[0], q.device

        out_q = F.normalize(self.encoder_q(q), dim=1)
        # out_q -> [batch_size, d_out]

        self._momentum_update_key_encoder()

        out_k = self.apply_encoder_k(k)
        # out_k -> [batch_size, d_out]

        logits_pos = torch.einsum(
            'b d, b d -> b', out_q, out_k)[:, None] / self.T
        # logits_pos -> [batch_size, 1]

        logits_neg = torch.einsum(
            'b d, k d -> b k', out_q, self.queue.clone().detach()) / self.T
        # logits_neg -> [batch_size, len_queue]

        # In case a single sample appears more than once in the queue,
        # mask out its negative logits
        if file_idxs is not None:
            mask = file_idxs[:, None] == self.queue_file
            # mask -> [batch_size, len_queue]
            logits_neg = logits_neg.masked_fill(mask, -float('inf'))

        idxs_hard = None
        enable1_hard = self.enable1_hard and step > self.start1_hard
        enable2_hard = self.enable2_hard and step > self.start2_hard
        if enable1_hard or enable2_hard:
            idxs_hard = self.find_hard_negatives(logits_neg)

        if enable1_hard:
            logits_neg = self.hard_negatives1(
                out_q=out_q, logits=logits_neg, idxs_hard=idxs_hard)

        if enable2_hard:
            logits_neg = self.hard_negatives2(
                out_q=out_q, logits=logits_neg, idxs_hard=idxs_hard)

        logits = torch.cat([logits_pos, logits_neg], dim=1)
        # logits -> [batch_size, len_queue + (s1_hard) + (s2_hard) + 1]

        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        # labels -> [batch_size]

        self._dequeue_and_enqueue(keys=out_k, file_idxs=file_idxs)

        return logits, labels
