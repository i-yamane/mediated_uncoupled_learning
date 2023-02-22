import torch
import torch.nn as nn


class MLP(nn.Module): # type: ignore
    # This does not seem to work properly.
    def __init__(
            self,
            dim_in: int,
            dim_hid: int,
            dim_out: int,
            n_layers: int=3,
            batch_norm: bool=False,
            output_probability: bool=False
        ) -> None:
        """ Multi-layer perceptron with `n_layers` layers.
        """
        super(MLP, self).__init__()
        self.output_probability = output_probability
        self.n_layers = n_layers
        self.batch_norm = batch_norm

        self.full = nn.ModuleList([nn.Linear(dim_in, dim_hid)])
        self.full.extend([nn.Linear(dim_hid, dim_hid) for _ in range(n_layers - 1)])

        if self.batch_norm:
            self.bnorm = nn.ModuleList([nn.BatchNorm1d(dim_hid) for _ in range(n_layers)])

        self.relu = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])

        self.full_out = nn.Linear(dim_hid, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """ Perform the forward calculation.
        """
        y = x

        for i in range(self.n_layers):
            y = self.full[i](y)
            # if self.batch_norm:
            #     y = self.bnorm[i](y)
            y = self.relu[i](y)

        y = self.full_out(y)
        if self.output_probability:
            y = torch.sigmoid(y)

        return y


class MLP3(nn.Module):  # type: ignore
    def __init__(
            self,
            dim_in: int,
            dim_hid: int,
            dim_out: int,
            output_probability: bool=False
        ) -> None:
        super(MLP3, self).__init__()

        self.output_probability = output_probability

        self.full1 = nn.Linear(dim_in, dim_hid)
        #self.bnorm1 = nn.BatchNorm1d(dim_hid)
        self.relu1 = nn.ReLU()

        self.full2 = nn.Linear(dim_hid, dim_hid)
        #self.bnorm2 = nn.BatchNorm1d(dim_hid)
        self.relu2 = nn.ReLU()

        self.full3 = nn.Linear(dim_hid, dim_hid)
        #self.bnorm3 = nn.BatchNorm1d(dim_hid)
        self.relu3 = nn.ReLU()

        self.fc_out = nn.Linear(dim_hid, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Forward calculation.
        """

        y = x

        y = self.full1(y)
        #y = self.bnorm1(y)
        y = self.relu1(y)

        y = self.full2(y)
        #y = self.bnorm2(y)
        y = self.relu2(y)

        y = self.full3(y)
        #y = self.bnorm3(y)
        y = self.relu3(y)

        y = self.fc_out(y)

        if self.output_probability:
            y = torch.sigmoid(y)

        return y


class LinearModel(nn.Module):  # type: ignore
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            output_probability: bool=False
        ) -> None:
        super(LinearModel, self).__init__()

        self.output_probability = output_probability

        self.fc_out = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Forward calculation.
        """

        y = self.fc_out(x)

        if self.output_probability:
            y = torch.sigmoid(y)

        return y
