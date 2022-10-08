"""
A GNN based model for decentralized controller
author: Xinchi Huang
"""
import torch
import torch.nn as nn
from model.weights_initializer import weights_init
from model.graphUtils import graphML as gml


class DecentralController(nn.Module):
    def __init__(self, number_of_agent=3, input_width=100, input_height=100,use_cuda=False):
        super().__init__()
        self.S = None
        self.numAgents = number_of_agent
        self.device = "cuda" if use_cuda else "cpu"
        conv_W=[input_width]
        conv_H=[input_height]
        action_num=2
        channel_num=[1] + [32, 32, 64, 64, 128]
        stride_num=[1, 1, 1, 1, 1]
        compress_MLP_dim = 1
        compress_features_num = [2 ** 7]

        max_pool_filter_Taps = 2
        max_pool_stride = 2
        # # 1 layer origin
        node_signals_dim = [2 ** 7]

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
        # self.F = [numCompressFeatures[-1]] + [numCompressFeatures[-1]] + [numCompressFeatures[-1]] + dimNodeSignals  # Features
        self.F = [compress_features_num[-1]] + node_signals_dim
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
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

        B = input_tensor.shape[0] # batch size
        # B x G x N
        extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(
            self.device
        )
        for id_agent in range(self.numAgents):
            # for id_agent in range(1):
            input_currentAgent = input_tensor[:, id_agent, :, :]
            input_currentAgent = input_currentAgent.unsqueeze(1).double()

            featureMap = self.ConvLayers(input_currentAgent)
            featureMapFlatten = featureMap.view(featureMap.size(0), -1)
            # extractFeatureMap[:, :, id_agent] = featureMapFlatten
            compressfeature = self.compressMLP(featureMapFlatten)
            extractFeatureMap[:, :, id_agent] = compressfeature  # B x F x N

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)
        sharedFeature = sharedFeature.permute(0, 2, 1)

        # ref angles and alpha concatenation
        for i in range(10):
            sharedFeature = torch.cat((sharedFeature, refs), dim=2)
        for i in range(10):
            sharedFeature = torch.cat((sharedFeature, alphas), dim=2)

        sharedFeature = sharedFeature.permute(0, 2, 1)
        sharedFeature = sharedFeature.float()
        action_predict = []
        for id_agent in range(self.numAgents):
            # for id_agent in range(1):
            # DCP_nonGCN
            # sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
            # DCP
            # torch.index_select(sharedFeature_currentAgent, 3, id_agent)
            sharedFeature_currentAgent = sharedFeature[:, :, id_agent]
            # print("sharedFeature_currentAgent.requires_grad: {}\n".format(sharedFeature_currentAgent.requires_grad))
            # print("sharedFeature_currentAgent.grad_fn: {}\n".format(sharedFeature_currentAgent.grad_fn))

            sharedFeatureFlatten = sharedFeature_currentAgent.view(
                sharedFeature_currentAgent.size(0), -1
            ).double()
            action_currentAgents = self.actionsMLP(sharedFeatureFlatten)  # 1 x 5
            action_predict.append(action_currentAgents)  # N x 5

        return action_predict
class DecentralPlannerNet(nn.Module):
    def __init__(self, nA=3, inW=100, inH=100):
        super().__init__()
        self.S = None
        self.numAgents = nA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        inW = inW
        inH = inH

        convW = [inW]
        convH = [inH]
        numAction = 2

        use_vgg = False

        # ------------------ DCP v1.4  -  with maxpool + non stride in CNN - less feature
        numChannel = [1] + [32, 32, 64, 64, 128]
        numStride = [1, 1, 1, 1, 1]

        dimCompressMLP = 1
        numCompressFeatures = [2**7]

        nMaxPoolFilterTaps = 2
        numMaxPoolStride = 2
        # # 1 layer origin
        dimNodeSignals = [2**7]

        # nGraphFilterTaps = [3,3,3]
        nGraphFilterTaps = [3]
        # --- actionMLP
        dimActionMLP = 3
        numActionFeatures = [numAction]

        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################
        convl = []
        numConv = len(numChannel) - 1
        nFilterTaps = [3] * numConv
        nPaddingSzie = [1] * numConv
        for l in range(numConv):
            convl.append(
                nn.Conv2d(
                    in_channels=numChannel[l],
                    out_channels=numChannel[l + 1],
                    kernel_size=nFilterTaps[l],
                    stride=numStride[l],
                    padding=nPaddingSzie[l],
                    bias=True,
                )
            )
            convl.append(nn.BatchNorm2d(num_features=numChannel[l + 1]))
            convl.append(nn.LeakyReLU(inplace=True))

            W_tmp = (
                int((convW[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l])
                + 1
            )
            H_tmp = (
                int((convH[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l])
                + 1
            )
            # Adding maxpooling
            if l % 2 == 0:
                convl.append(nn.MaxPool2d(kernel_size=2))
                W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                # http://cs231n.github.io/convolutional-networks/
            convW.append(W_tmp)
            convH.append(H_tmp)

        self.ConvLayers = nn.Sequential(*convl)

        numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]

        #####################################################################
        #                                                                   #
        #                MLP-feature compression                            #
        #                                                                   #
        #####################################################################

        numCompressFeatures = [numFeatureMap] + numCompressFeatures

        compressmlp = []
        for l in range(dimCompressMLP):
            compressmlp.append(
                nn.Linear(
                    in_features=numCompressFeatures[l],
                    out_features=numCompressFeatures[l + 1],
                    bias=True,
                )
            )
            compressmlp.append(nn.LeakyReLU(inplace=True))

        self.compressMLP = nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        # self.F = [numCompressFeatures[-1]] + [numCompressFeatures[-1]] + [numCompressFeatures[-1]] + dimNodeSignals  # Features
        self.F = [numCompressFeatures[-1]] + dimNodeSignals
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
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
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        # + 10 for ref angles, +10 for alphas
        numActionFeatures = (
            [self.F[-1] + 10 + 10] + [self.F[-1]] + [self.F[-1]] + numActionFeatures
        )
        actionsfc = []
        for l in range(dimActionMLP):
            if l < (dimActionMLP - 1):
                actionsfc.append(
                    nn.Linear(
                        in_features=numActionFeatures[l],
                        out_features=numActionFeatures[l + 1],
                        bias=True,
                    )
                )
                actionsfc.append(nn.LeakyReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(
                        in_features=numActionFeatures[l],
                        out_features=numActionFeatures[l + 1],
                        bias=True,
                    )
                )

        self.actionsMLP = nn.Sequential(*actionsfc)
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

    def forward(self, inputTensor, refs, alphas):

        B = inputTensor.shape[0]  # batch size
        # B x G x N
        extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(
            self.device
        )
        for id_agent in range(self.numAgents):
            # for id_agent in range(1):
            input_current_agent = inputTensor[:, id_agent, :, :]
            input_current_agent = input_current_agent.unsqueeze(1).double()
            featureMap = self.ConvLayers(input_current_agent)
            featureMapFlatten = featureMap.view(featureMap.size(0), -1)
            # extractFeatureMap[:, :, id_agent] = featureMapFlatten
            compressfeature = self.compressMLP(featureMapFlatten)
            extractFeatureMap[:, :, id_agent] = compressfeature  # B x F x N

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)
        sharedFeature = sharedFeature.permute(0, 2, 1)

        # ref angles and alpha concatenation
        for i in range(10):
            sharedFeature = torch.cat((sharedFeature, refs), dim=2)
        for i in range(10):
            sharedFeature = torch.cat((sharedFeature, alphas), dim=2)

        sharedFeature = sharedFeature.permute(0, 2, 1)
        sharedFeature = sharedFeature.float()
        action_predict = []
        for id_agent in range(self.numAgents):
            # for id_agent in range(1):
            # DCP_nonGCN
            # sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
            # DCP
            # torch.index_select(sharedFeature_currentAgent, 3, id_agent)
            sharedFeature_currentAgent = sharedFeature[:, :, id_agent]
            # print("sharedFeature_currentAgent.requires_grad: {}\n".format(sharedFeature_currentAgent.requires_grad))
            # print("sharedFeature_currentAgent.grad_fn: {}\n".format(sharedFeature_currentAgent.grad_fn))

            sharedFeatureFlatten = sharedFeature_currentAgent.view(
                sharedFeature_currentAgent.size(0), -1
            ).double()
            action_currentAgents = self.actionsMLP(sharedFeatureFlatten)  # 1 x 5
            action_predict.append(action_currentAgents)  # N x 5

        return action_predict

    def forward_one(self, inputTensor, refs, alphas):

        """

        :param inputTensor: Concatenated occupancy maps
        :param refs: zeros(useless)
        :param alphas: formation scale(default=2)
        :return:
        """

        B = inputTensor.shape[0]  # batch size
        # B x G x N
        extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(
            self.device
        )
        for id_agent in range(1):
            input_currentAgent = inputTensor[:, id_agent, :, :]
            input_currentAgent = input_currentAgent.unsqueeze(1).double()
            featureMap = self.ConvLayers(input_currentAgent)
            featureMapFlatten = featureMap.view(featureMap.size(0), -1)
            # extractFeatureMap[:, :, id_agent] = featureMapFlatten
            compressfeature = self.compressMLP(featureMapFlatten)
            extractFeatureMap[:, :, id_agent] = compressfeature  # B x F x N

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)
        sharedFeature = sharedFeature.permute(0, 2, 1)
        for i in range(10):
            sharedFeature = torch.cat((sharedFeature, refs), dim=2)
        for i in range(10):
            sharedFeature = torch.cat((sharedFeature, alphas), dim=2)
        sharedFeature = sharedFeature.permute(0, 2, 1)
        action_predict = []
        for id_agent in range(1):
            # DCP_nonGCN
            # sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
            # DCP
            # torch.index_select(sharedFeature_currentAgent, 3, id_agent)
            sharedFeature_currentAgent = sharedFeature[:, :, id_agent]
            # print("sharedFeature_currentAgent.requires_grad: {}\n".format(sharedFeature_currentAgent.requires_grad))
            # print("sharedFeature_currentAgent.grad_fn: {}\n".format(sharedFeature_currentAgent.grad_fn))

            sharedFeatureFlatten = sharedFeature_currentAgent.view(
                sharedFeature_currentAgent.size(0), -1
            )
            action_currentAgents = self.actionsMLP(sharedFeatureFlatten)  # 1 x 5
            action_predict.append(action_currentAgents)  # N x 5

        return action_predict
