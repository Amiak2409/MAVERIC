import torch
import torch.nn as nn
import torch.optim as optim
from FollowingDistancePredictor import FollowingDistancePredictor
from LaneChangePredictor import LaneChangePredictor
from VelocityPredictor import VelocityPredictor
from StylePredictor import StylePredictor
from MutualInformation import MutualInformation

class MAVERIC(nn.Module):
    def __init__(self, embed_dim, f_dim, l_dim, v_dim, mi_dim):
        super(MAVERIC, self).__init__()
        self.embed_dim = embed_dim
        
        self.embed = nn.Parameter(torch.rand(1, embed_dim), requires_grad=True)
        self.following_distance = FollowingDistancePredictor(f_dim, embed_dim)
        self.lane_change = LaneChangePredictor(l_dim, embed_dim)
        self.velocity = VelocityPredictor(v_dim, embed_dim)
        self.style = StylePredictor(embed_dim)
        self.mutual_information = MutualInformation(mi_dim, embed_dim)
        
        self.optimizer_embed = optim.SGD([self.embed], lr=0.01, momentum=0.9)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, f_state, l_state, v_state):
        embed = self.embed
        f_pred = self.following_distance(f_state, embed)
        ml_state, l_pred = self.lane_change(l_state, embed)
        mv_state, v_pred = self.velocity(v_state, embed)
        s_pred = self.style(embed)
        
        X = torch.cat((ml_state, mv_state, f_state), dim=1)
        P = torch.cat((f_pred, l_pred, v_pred, s_pred), dim=1)
        
        mi_pred = self.mutual_information(X, P)
        
        self.embed = mi_pred
        
        return f_pred, l_pred, v_pred, s_pred, mi_pred


    def train1(self, f_state, l_state, v_state, X, P, labels):
        self.optimizer_embed.zero_grad()

        f_loss = self.following_distance.train(f_state, self.embed, labels[0])
        l_loss = self.line_change.train(l_state, self.embed, labels[1])
        v_loss = self.velocity.train(v_state, self.embed, labels[2])
        s_loss = self.style.train(self.embed, labels[3])
        mi_loss = self.mutual_information.train(X, P, self.embed)
        
        loss = f_loss + l_loss + v_loss + s_loss + mi_loss
        
        loss.backward()
        self.optimizer_embed.step()
        
        print(f"loss1: {loss}")
        
    
    def train2(self, f_state, l_state, v_state, X, P, labels):
        self.optimizer.zero_grad()

        f_input = torch.cat((f_state, self.embed), dim=1)
        l_input = torch.cat((l_state, self.embed), dim=1)
        v_input = torch.cat((v_state, self.embed), dim=1)
        s_input = self.embed
        mi_input = torch.cat((X, P), dim=1)
        
        f_loss = self.following_distance.criterion(f_input, labels[0])
        l_loss = self.following_distance.criterion(l_input, labels[1])
        v_loss = self.following_distance.criterion(v_input, labels[2])
        s_loss = self.following_distance.criterion(s_input, labels[3])
        mi_loss = self.following_distance.criterion(mi_input, self.embed)
        

        loss = f_loss + l_loss + v_loss + s_loss + mi_loss
        
        loss.backward()
        self.optimizer.step()
        
        print(f"loss2: {loss}")
    
        
    def apply_to_you(self, f_state, l_state, v_state, X, P, labels):
        self.optimizer_embed.zero_grad()

        f_input = torch.cat((f_state, self.embed), dim=1)
        l_input = torch.cat((l_state, self.embed), dim=1)
        v_input = torch.cat((v_state, self.embed), dim=1)
        s_input = self.embed
        mi_input = torch.cat((X, P), dim=1)
        
        f_loss = self.following_distance.criterion(f_input, labels[0])
        l_loss = self.following_distance.criterion(l_input, labels[1])
        v_loss = self.following_distance.criterion(v_input, labels[2])
        s_loss = self.following_distance.criterion(s_input, labels[3])
        mi_loss = self.following_distance.criterion(mi_input, self.embed)
        

        loss = f_loss + l_loss + v_loss + s_loss + mi_loss
        
        loss.backward()
        self.optimizer_embed.step()
        
        print(f"loss2: {loss}")



