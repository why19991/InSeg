def FCA(labels=None, features=None, centroids=None, unknown_label=None, features_counter=None,strategy='running'):
    b, h, w = labels.shape
    labels_down = labels.unsqueeze(dim=1)
    cl_present = torch.unique(input=labels_down)
    for cl in cl_present:
        if cl > 0 and cl != 255:
            features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(-1, features.shape[1])
            if strategy == 'running':
                features_counter[cl] = features_counter[cl] + features_cl.shape[0]
                centroids[cl] = (centroids[cl] * features_counter[cl] + features_cl.detach().sum(dim=0)) / features_counter[cl]
            else:
                centroids[cl] = centroids[cl] * 0.999 + (1 - 0.999) * features_cl.detach().mean(dim=0)
    centroids = F.normalize(centroids, p=2, dim=1)
    features_bg = features[(labels_down == 0).expand(-1, features.shape[1], -1, -1)].view(-1, features.shape[1])
    features_bg = F.normalize(features_bg, p=2, dim=1)
    re_labels = labels_down.view(-1)
    similarity = torch.matmul(features_bg.detach(), centroids.T)
    similarity = similarity.mean(dim=-1)
    value, index = torch.sort(similarity, descending=True)
    fill_mask = torch.zeros_like(index)
    value_index = value >= 0.8
    fill_index = index[value_index]
    fill_mask[fill_index] = unknown_label # we set the label of unknown class as C_t+1  in our experiment
    re_labels[re_labels==0] = fill_mask
    re_labels = re_labels.view(b, h , w )
    return re_labels, centroids

def CID(outputs=None, outputs_old=None, nb_old_classes=None, nb_current_classes=None, nb_future_classes=None, labels=None):

    outputs = outputs.permute(0, 2, 3, 1).contiguous()
    b, h, w, c = outputs.shape
    outputs_old = outputs_old.permute(0, 2, 3, 1).contiguous()
    out_old = torch.zeros_like(outputs)
    labels_unique = torch.unique(labels)
    out_old[..., :nb_old_classes + nb_future_classes] = outputs_old[..., :]
    for cl in range(nb_old_classes, nb_current_classes):
        out_old[..., cl] = outputs_old[..., 0] * (labels==cl).squeeze(dim=-1)
        for j in range(nb_future_classes):
            out_old[..., cl] = out_old[..., cl] + outputs_old[..., nb_old_classes + j] * (labels==cl).squeeze(dim=-1)
            # out_old[..., nb_old_classes+j] = out_old[..., cl] + outputs_old[..., nb_old_classes + j] * (labels == cl).squeeze(dim=-1)
        out_old[..., 0] = (labels != cl).squeeze(dim=-1) * out_old[..., 0]
    # out_old[..., :nb_old_classes+nb_future_classes] = outputs_old[..., :]* ((labels < nb_old_classes) + (labels == 255)).unsqueeze(dim=-1)
    out_old = torch.log_softmax(out_old, dim=-1)
    outputs = torch.softmax(outputs, dim=-1)
    # out = (out_old * outputs * ((labels<nb_current_classes) + (labels==255)).unsqueeze(dim=-1).expand(-1, -1, -1, c)).sum(dim=-1) / c

    out = (out_old * outputs * ((labels<nb_old_classes) + (labels==255)).unsqueeze(dim=-1).expand(-1, -1, -1, c)).sum(dim=-1) / c + \
          (out_old * outputs * ((labels < nb_current_classes) * (labels >= nb_old_classes)).unsqueeze(dim=-1).expand(-1, -1, -1,c)).sum(dim=-1) / c

    return - torch.mean(out)



