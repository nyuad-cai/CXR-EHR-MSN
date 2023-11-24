import math
import torch

import torch.nn as nn
import torch.nn.functional as F



def prototype_probabilities(queries: torch.Tensor,
                            prototypes: torch.Tensor,
                            temperature: float,
                            ) -> torch.Tensor:
    """Returns probability for each query to belong to each prototype.

    Args:
        queries:
            Tensor with shape (batch_size, dim), projection head output
        prototypes:
            Tensor with shape (num_prototypes, dim)
        temperature:
            Inverse scaling factor for the similarity.

    Returns:
        Probability tensor with shape (batch_size, num_prototypes) which sums to 1 along
        the num_prototypes dimension.

    """                           
    return F.softmax(torch.matmul(queries, prototypes.T) / temperature, dim=1)


def sharpen(probabilities: torch.Tensor, 
            temperature: float
           ) -> torch.Tensor:
    """Sharpens the probabilities with the given temperature.

    Args:
        probabilities:
            Tensor with shape (batch_size, dim)
        temperature:
            Temperature in (0, 1]. Lower temperature results in stronger sharpening (
            output probabilities are less uniform).
    Returns:
        Probabilities tensor with shape (batch_size, num_prototypes).

    """
    probabilities = probabilities ** (1.0 / temperature)
    probabilities /= torch.sum(probabilities, dim=1, keepdim=True)
    return probabilities


@torch.no_grad()
def sinkhorn(probabilities: torch.Tensor,
             iterations: int = 3,
             gather_distributed: bool = False,
            ) -> torch.Tensor:
    """Runs sinkhorn normalization on the probabilities as described in [0].

    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Args:
        probabilities:
            Probabilities tensor with shape (batch_size, num_prototypes).
        iterations:
            Number of iterations of the sinkhorn algorithms. Set to 0 to disable.
        gather_distributed:
            If True then features from all gpus are gathered during normalization.
    Returns:
        A normalized probabilities tensor.

    """
    if iterations <= 0:
        return probabilities


    num_targets, num_prototypes = probabilities.shape
    probabilities = probabilities.T
    sum_probabilities = torch.sum(probabilities)

    probabilities = probabilities / sum_probabilities

    for _ in range(iterations):
        # normalize rows
        row_sum = torch.sum(probabilities, dim=1, keepdim=True)

        probabilities /= row_sum
        probabilities /= num_prototypes

        # normalize columns
        probabilities /= torch.sum(probabilities, dim=0, keepdim=True)
        probabilities /= num_targets

    probabilities *= num_targets
    return probabilities.T


def regularization_loss(mean_anchor_probs: torch.Tensor
                       ) -> torch.Tensor:
    """Calculates mean entropy regularization loss."""
    loss = -torch.sum(torch.log(mean_anchor_probs ** (-mean_anchor_probs)))
    loss += math.log(float(len(mean_anchor_probs)))
    return loss    



class MSNLoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.1,
                 sinkhorn_iterations: int = 0,
                 similarity_weight: float = 1.0,
                 age_weight: float = 1.0,
                 gender_weight: float = 1.0,
                 regularization_weight: float = 1.0,
                 gather_distributed: bool = False,
               ) -> None:

        super().__init__()
        
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.similarity_weight = similarity_weight
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.regularization_weight = regularization_weight

        
    def forward(self,
                anchors: torch.Tensor,
                targets: torch.Tensor,
                prototypes: torch.Tensor,
                target_sharpen_temperature: float = 0.25,
                focal: bool = True
                      ) -> torch.Tensor:
        
        similarity_loss = self.similarity_weight * self._forward_loss(anchors=anchors[0] if focal else anchors[:,0],
                                                                      targets=targets[:,0],
                                                                      prototypes=prototypes[0].weight.data
                                                                     )

        age_loss = self.age_weight * self._forward_loss(anchors=anchors[1] if focal else anchors[:,1],
                                                           targets=targets[:,1],
                                                           prototypes=prototypes[1].weight.data
                                                          )

        gender_loss = self.gender_weight * self._forward_loss(anchors=anchors[2] if focal else anchors[:,2],
                                                              targets=targets[:,2],
                                                              prototypes=prototypes[2].weight.data
                                                             )

        loss = similarity_loss + age_loss + gender_loss
        return loss
    
    def _forward_loss(self,
                anchors: torch.Tensor,
                targets: torch.Tensor,
                prototypes: torch.Tensor,
                target_sharpen_temperature: float = 0.25,
               ) -> torch.Tensor:

        num_views = anchors.shape[0] // targets.shape[0]
        anchors = F.normalize(anchors, dim=1)
        targets = F.normalize(targets, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        anchor_probs = prototype_probabilities(anchors, 
                                               prototypes, 
                                               temperature=self.temperature
                                              )

        with torch.no_grad():
            target_probs = prototype_probabilities(targets, 
                                                   prototypes, 
                                                   temperature=self.temperature
                                                   )
            target_probs = sharpen(target_probs, temperature=target_sharpen_temperature)
            if self.sinkhorn_iterations > 0:
                target_probs = sinkhorn(probabilities=target_probs,
                                        iterations=self.sinkhorn_iterations,
                                        )
            target_probs = torch.repeat_interleave(target_probs, repeats=num_views, dim=0)
        
        loss = torch.mean(torch.sum(torch.log(anchor_probs ** (-target_probs)), dim=1))

        # regularization loss
        if self.regularization_weight > 0:
            mean_anchor_probs = torch.mean(anchor_probs, dim=0)
            reg_loss = regularization_loss(mean_anchor_probs=mean_anchor_probs)
            loss += self.regularization_weight * reg_loss
         
        return loss
    

