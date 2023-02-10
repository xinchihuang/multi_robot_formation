"""
A GNN based model for decentralized controller
author: Xinchi Huang
"""
import torch
import torch.nn as nn
from weights_initializer import weights_init
from graphUtils import graphML as gml


class GnnMapBasic(nn.Module):
    def __init__(
        self, number_of_agent=3, input_width=100, input_height=100, use_cuda=False
    ):
        print("using GNN model controller (map)")
        super().__init__()
        self.S = None
        self.number_of_agent = number_of_agent
        self.device = "cuda" if use_cuda else "cpu"
        conv_W = [input_width]
        conv_H = [input_height]
        action_num = 2
        channel_num = [1] + [32, 32, 64, 64, 128]
        stride_num = [1, 1, 1, 1, 1]
        compress_MLP_dim = 1
        compress_features_num = [2**7]

        max_pool_filter_Taps = 2
        max_pool_stride = 2
        # # 1 layer origin
        node_signals_dim = [2**7]

        # nGraphFilterTaps = [3,3,3]
        graph_filter_taps_num = [3]
        # --- actionMLP
        action_MLP_dim = 3
        action_features_num = [action_num]

        #####################################################################
        #                                                                   #
        #                CNN layers to extract feature                      #
        #                                                                   #
        #####################################################################

        conv_layers = []
        conv_num = len(channel_num) - 1
        filter_taps = [3] * conv_num
        padding_size = [1] * conv_num
        for l in range(conv_num):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=channel_num[l],
                    out_channels=channel_num[l + 1],
                    kernel_size=filter_taps[l],
                    stride=stride_num[l],
                    padding=padding_size[l],
                    bias=True,
                )
            )
            conv_layers.append(nn.BatchNorm2d(num_features=channel_num[l + 1]))
            conv_layers.append(nn.LeakyReLU(inplace=True))

            W_tmp = (
                int((conv_W[l] - filter_taps[l] + 2 * padding_size[l]) / stride_num[l])
                + 1
            )
            H_tmp = (
                int((conv_H[l] - filter_taps[l] + 2 * padding_size[l]) / stride_num[l])
                + 1
            )
            # Adding maxpooling
            if l % 2 == 0:
                conv_layers.append(nn.MaxPool2d(kernel_size=2))
                W_tmp = int((W_tmp - max_pool_filter_Taps) / max_pool_stride) + 1
                H_tmp = int((H_tmp - max_pool_filter_Taps) / max_pool_stride) + 1
                # http://cs231n.github.io/convolutional-networks/
            conv_W.append(W_tmp)
            conv_H.append(H_tmp)

        self.ConvLayers = nn.Sequential(*conv_layers).double()

        num_feature_map = channel_num[-1] * conv_W[-1] * conv_H[-1]
        #####################################################################
        #                                                                   #
        #                MLP-feature compression                            #
        #                                                                   #
        #####################################################################

        compress_features_num = [num_feature_map] + compress_features_num

        compress_mlp = []
        for l in range(compress_MLP_dim):
            compress_mlp.append(
                nn.Linear(
                    in_features=compress_features_num[l],
                    out_features=compress_features_num[l + 1],
                    bias=True,
                )
            )
            compress_mlp.append(nn.LeakyReLU(inplace=True))

        self.compressMLP = nn.Sequential(*compress_mlp).double()

        self.numFeatures2Share = compress_features_num[-1]
        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(graph_filter_taps_num)  # Number of graph filtering layers
        self.F = [compress_features_num[-1]] + node_signals_dim
        self.K = graph_filter_taps_num  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(
                gml.GraphFilterBatch(
                    self.F[l], self.F[l + 1], self.K[l], self.E, self.bias
                )
            )
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            gfl.append(nn.LeakyReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl).double()  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        # + 10 for ref angles, +10 for alphas
        action_features_num = (
            [self.F[-1] + 10 + 10] + [self.F[-1]] + [self.F[-1]] + action_features_num
        )
        actionsfc = []
        for l in range(action_MLP_dim):
            if l < (action_MLP_dim - 1):
                actionsfc.append(
                    nn.Linear(
                        in_features=action_features_num[l],
                        out_features=action_features_num[l + 1],
                        bias=True,
                    )
                )
                actionsfc.append(nn.LeakyReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(
                        in_features=action_features_num[l],
                        out_features=action_features_num[l + 1],
                        bias=True,
                    )
                )

        self.actionsMLP = nn.Sequential(*actionsfc).double()
        self.apply(weights_init)

    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self, input_tensor, refs, alphas):

        B = input_tensor.shape[0]  # batch size
        # B x G x N
        extract_feature_map = torch.zeros(
            B, self.numFeatures2Share, self.number_of_agent
        ).to(self.device)
        for id_agent in range(self.number_of_agent):
            # for id_agent in range(1):
            input_current_agent = input_tensor[:, id_agent, :, :]
            input_current_agent = input_current_agent.unsqueeze(1).double()

            feature_map = self.ConvLayers(input_current_agent)
            feature_map_flatten = feature_map.view(feature_map.size(0), -1)
            # extract_feature_map[:, :, id_agent] = feature_mapFlatten

            compressed_feature = self.compressMLP(feature_map_flatten)
            extract_feature_map[:, :, id_agent] = compressed_feature  # B x F x N



        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        shared_feature = self.GFL(extract_feature_map)
        shared_feature = shared_feature.permute(0, 2, 1)

        # ref angles and alpha concatenation
        for i in range(10):
            shared_feature = torch.cat((shared_feature, refs), dim=2)
        for i in range(10):
            shared_feature = torch.cat((shared_feature, alphas), dim=2)

        shared_feature = shared_feature.permute(0, 2, 1)
        shared_feature = shared_feature.float()
        action_predict = []
        for id_agent in range(self.number_of_agent):
            # for id_agent in range(1):
            # DCP_nonGCN
            # shared_feature_currentAgent = extract_feature_map[:, :, id_agent]
            # DCP
            # torch.index_select(shared_feature_currentAgent, 3, id_agent)
            shared_feature_current = shared_feature[:, :, id_agent]

            shared_feature_flatten = shared_feature_current.view(
                shared_feature_current.size(0), -1
            ).double()
            action_current = self.actionsMLP(shared_feature_flatten)  # 1 x 5
            action_predict.append(action_current)  # N x 5

        return action_predict
class GnnMapDecentralized(nn.Module):
    def __init__(
        self, number_of_agent=3, input_width=100, input_height=100, use_cuda=False
    ):
        print("using GNN model controller (map no communication)")
        super().__init__()
        self.S = None
        self.number_of_agent = number_of_agent
        self.device = "cuda" if use_cuda else "cpu"
        conv_W = [input_width]
        conv_H = [input_height]
        action_num = 2
        channel_num = [1] + [32, 32, 64, 64, 128]
        stride_num = [1, 1, 1, 1, 1]
        compress_MLP_dim = 1
        compress_features_num = [2**7]

        max_pool_filter_Taps = 2
        max_pool_stride = 2
        # # 1 layer origin
        node_signals_dim = [2**7]

        # nGraphFilterTaps = [3,3,3]
        graph_filter_taps_num = [3]
        # --- actionMLP
        action_MLP_dim = 3
        action_features_num = [action_num]

        #####################################################################
        #                                                                   #
        #                CNN layers to extract feature                      #
        #                                                                   #
        #####################################################################

        conv_layers = []
        conv_num = len(channel_num) - 1
        filter_taps = [3] * conv_num
        padding_size = [1] * conv_num
        for l in range(conv_num):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=channel_num[l],
                    out_channels=channel_num[l + 1],
                    kernel_size=filter_taps[l],
                    stride=stride_num[l],
                    padding=padding_size[l],
                    bias=True,
                )
            )
            conv_layers.append(nn.BatchNorm2d(num_features=channel_num[l + 1]))
            conv_layers.append(nn.LeakyReLU(inplace=True))

            W_tmp = (
                int((conv_W[l] - filter_taps[l] + 2 * padding_size[l]) / stride_num[l])
                + 1
            )
            H_tmp = (
                int((conv_H[l] - filter_taps[l] + 2 * padding_size[l]) / stride_num[l])
                + 1
            )
            # Adding maxpooling
            if l % 2 == 0:
                conv_layers.append(nn.MaxPool2d(kernel_size=2))
                W_tmp = int((W_tmp - max_pool_filter_Taps) / max_pool_stride) + 1
                H_tmp = int((H_tmp - max_pool_filter_Taps) / max_pool_stride) + 1
                # http://cs231n.github.io/convolutional-networks/
            conv_W.append(W_tmp)
            conv_H.append(H_tmp)

        self.ConvLayers = nn.Sequential(*conv_layers).double()

        num_feature_map = channel_num[-1] * conv_W[-1] * conv_H[-1]
        #####################################################################
        #                                                                   #
        #                MLP-feature compression                            #
        #                                                                   #
        #####################################################################

        compress_features_num = [num_feature_map] + compress_features_num

        compress_mlp = []
        for l in range(compress_MLP_dim):
            compress_mlp.append(
                nn.Linear(
                    in_features=compress_features_num[l],
                    out_features=compress_features_num[l + 1],
                    bias=True,
                )
            )
            compress_mlp.append(nn.LeakyReLU(inplace=True))

        self.compressMLP = nn.Sequential(*compress_mlp).double()

        self.numFeatures2Share = compress_features_num[-1]
        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(graph_filter_taps_num)  # Number of graph filtering layers
        self.F = [compress_features_num[-1]] + node_signals_dim
        self.K = graph_filter_taps_num  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(
                gml.GraphFilterBatch(
                    self.F[l], self.F[l + 1], self.K[l], self.E, self.bias
                )
            )
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            gfl.append(nn.LeakyReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl).double()  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        # + 10 for ref angles, +10 for alphas
        action_features_num = (
            [self.F[-1] + 10 + 10] + [self.F[-1]] + [self.F[-1]] + action_features_num
        )
        actionsfc = []
        for l in range(action_MLP_dim):
            if l < (action_MLP_dim - 1):
                actionsfc.append(
                    nn.Linear(
                        in_features=action_features_num[l],
                        out_features=action_features_num[l + 1],
                        bias=True,
                    )
                )
                actionsfc.append(nn.LeakyReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(
                        in_features=action_features_num[l],
                        out_features=action_features_num[l + 1],
                        bias=True,
                    )
                )

        self.actionsMLP = nn.Sequential(*actionsfc).double()
        self.apply(weights_init)

    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self, self_input_tensor,outer_msg,self_id, refs, alphas):

        B = self_input_tensor.shape[0]  # batch size
        # B x G x N
        extract_feature_map = torch.zeros(
            B, self.numFeatures2Share, self.number_of_agent
        ).to(self.device)
            # for id_agent in range(1):

        input_current_agent = self_input_tensor[:, self_id, :, :]
        input_current_agent = input_current_agent.unsqueeze(1).double()

        feature_map = self.ConvLayers(input_current_agent)
        feature_map_flatten = feature_map.view(feature_map.size(0), -1)
        # extract_feature_map[:, :, id_agent] = feature_mapFlatten

        compressed_feature = self.compressMLP(feature_map_flatten)
        extract_feature_map[:, :, self_id] = compressed_feature  # B x F x N


        for id,outer_feature in outer_msg.items():
            extract_feature_map[:, :, id]=outer_feature


        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        shared_feature = self.GFL(extract_feature_map)
        shared_feature = shared_feature.permute(0, 2, 1)

        # ref angles and alpha concatenation
        for i in range(10):
            shared_feature = torch.cat((shared_feature, refs), dim=2)
        for i in range(10):
            shared_feature = torch.cat((shared_feature, alphas), dim=2)

        shared_feature = shared_feature.permute(0, 2, 1)
        shared_feature = shared_feature.float()
        action_predict = []
        for id_agent in range(self.number_of_agent):
            # for id_agent in range(1):
            # DCP_nonGCN
            # shared_feature_currentAgent = extract_feature_map[:, :, id_agent]
            # DCP
            # torch.index_select(shared_feature_currentAgent, 3, id_agent)
            shared_feature_current = shared_feature[:, :, id_agent]

            shared_feature_flatten = shared_feature_current.view(
                shared_feature_current.size(0), -1
            ).double()
            action_current = self.actionsMLP(shared_feature_flatten)  # 1 x 5
            action_predict.append(action_current)  # N x 5

        return action_predict



class GnnPoseBasic(nn.Module):
    def __init__(self, number_of_agent=3, use_cuda=False):

        super().__init__()
        print("using GNN model controller (Pose)")
        self.S = None
        self.number_of_agent = number_of_agent
        self.device = "cuda" if use_cuda else "cpu"
        action_num = 2
        compress_features_num = [2**5]
        node_signals_dim = [2**5]

        # nGraphFilterTaps = [3,3,3]
        graph_filter_taps_num = [3]
        # --- actionMLP
        action_MLP_dim = 3
        action_features_num = [action_num]

        #####################################################################
        #                                                                   #
        #                MLP-feature extraction                            #
        #                                                                   #
        #####################################################################

        compress_mlp = []
        compress_mlp.append(
            nn.Linear(
                in_features=3,
                out_features=32,
                bias=True,
            )
        )
        compress_mlp.append(nn.LeakyReLU(inplace=True))
        compress_mlp.append(
            nn.Linear(
                in_features=32,
                out_features=32,
                bias=True,
            )
        )
        compress_mlp.append(nn.LeakyReLU(inplace=True))
        compress_mlp.append(
            nn.Linear(
                in_features=32,
                out_features=32,
                bias=True,
            )
        )

        # compress_mlp.append(
        #     nn.Linear(
        #         in_features=128,
        #         out_features=128,
        #         bias=True,
        #     )
        # )

        self.compressMLP = nn.Sequential(*compress_mlp).double()

        self.numFeatures2Share = compress_features_num[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(graph_filter_taps_num)  # Number of graph filtering layers
        self.F = [compress_features_num[-1]] + node_signals_dim
        self.K = graph_filter_taps_num  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(
                gml.GraphFilterBatch(
                    self.F[l], self.F[l + 1], self.K[l], self.E, self.bias
                )
            )
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            gfl.append(nn.LeakyReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl).double()  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        # + 10 for ref angles, +10 for alphas
        action_features_num = (
            [self.F[-1] + 10 + 10] + [self.F[-1]] + [self.F[-1]] + action_features_num
        )
        actionsfc = []
        for l in range(action_MLP_dim):
            if l < (action_MLP_dim - 1):
                actionsfc.append(
                    nn.Linear(
                        in_features=action_features_num[l],
                        out_features=action_features_num[l + 1],
                        bias=True,
                    )
                )
                actionsfc.append(nn.LeakyReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(
                        in_features=action_features_num[l],
                        out_features=action_features_num[l + 1],
                        bias=True,
                    )
                )

        self.actionsMLP = nn.Sequential(*actionsfc).double()
        self.apply(weights_init)

    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self, input_tensor, refs, alphas):

        B = input_tensor.shape[0]  # batch size
        # B x G x N
        extract_feature_map = torch.zeros(
            B, self.numFeatures2Share, self.number_of_agent
        ).to(self.device)
        for id_agent in range(self.number_of_agent):
            position_input = input_tensor[:, id_agent, :, :]
            compressed_feature = self.compressMLP(position_input)
            compressed_feature = torch.sum(compressed_feature, 1)

            extract_feature_map[:, :, id_agent] = compressed_feature  # B x F x N




        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        shared_feature = self.GFL(extract_feature_map)
        shared_feature = shared_feature.permute(0, 2, 1)

        # ref angles and alpha concatenation
        for i in range(10):
            shared_feature = torch.cat((shared_feature, refs), dim=2)
        for i in range(10):
            shared_feature = torch.cat((shared_feature, alphas), dim=2)

        shared_feature = shared_feature.permute(0, 2, 1)
        shared_feature = shared_feature.float()
        action_predict = []
        for id_agent in range(self.number_of_agent):
            # for id_agent in range(1):
            # DCP_nonGCN
            # shared_feature_currentAgent = extract_feature_map[:, :, id_agent]
            # DCP
            # torch.index_select(shared_feature_currentAgent, 3, id_agent)
            shared_feature_current = shared_feature[:, :, id_agent]

            shared_feature_flatten = shared_feature_current.view(
                shared_feature_current.size(0), -1
            ).double()
            action_current = self.actionsMLP(shared_feature_flatten)  # 1 x 5
            action_predict.append(action_current)  # N x 5

        return action_predict

class DummyModel(nn.Module):
    def __init__(self, number_of_agent=3, use_cuda=False):
        print("using dummy model controller")
        self.use_cuda = use_cuda
        self.number_of_agent = number_of_agent
        self.device = "cuda" if use_cuda else "cpu"
        super().__init__()
        mlp = []
        mlp.append(
            nn.Linear(
                in_features=3,
                out_features=128,
                bias=True,
            )
        )
        mlp.append(nn.LeakyReLU(inplace=True))
        mlp.append(
            nn.Linear(
                in_features=128,
                out_features=128,
                bias=True,
            )
        )
        mlp.append(nn.LeakyReLU(inplace=True))
        mlp.append(
            nn.Linear(
                in_features=128,
                out_features=2,
                bias=True,
            )
        )
        mlp.append(nn.LeakyReLU(inplace=True))
        self.MLP = nn.Sequential(*mlp).double()
        self.out_put = nn.Linear(
            in_features=2,
            out_features=2,
            bias=True,
        ).double()

        self.apply(weights_init)
        # nn.Dropout(0.25)

    def forward(self, input_tensor):
        action_predict = []
        for id_agent in range(self.number_of_agent):
            position_input = input_tensor[:, id_agent, :, :]
            features = self.MLP(position_input)
            features = torch.sum(features, 1)
            action_current = self.out_put(features)
            action_predict.append(action_current)
        return action_predict
