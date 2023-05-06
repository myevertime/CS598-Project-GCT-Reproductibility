import torch
import numpy as np
import os
import sys
import math
import logging
import json
from tqdm import tqdm, trange
from process_eicu import get_datasets
from utils import *
from graph_convolutional_transformer import GraphConvolutionalTransformer

from tensorboardX import SummaryWriter
import torchsummary as summary


def prediction_loop(args, model, dataloader, priors_datalaoder, description='Evaluating'):
    batch_size = dataloader.batch_size
    eval_losses = []
    preds = None
    label_ids = None
    model.eval()

    for data, priors_data in tqdm(zip(dataloader, priors_datalaoder), desc=description):
        data, priors_data = prepare_data(data, priors_data, args.device)
        with torch.no_grad():
            outputs = model(data, priors_data)
            loss = outputs[0].mean().item()
            logits = outputs[1]

        labels = data[args.label_key]

        batch_size = data[list(data.keys())[0]].shape[0]
        eval_losses.extend([loss] * batch_size)
        preds = logits if preds is None else nested_concat(preds, logits, dim=0)
        label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

    if preds is not None:
        preds = nested_numpify(preds)
    if label_ids is not None:
        label_ids = nested_numpify(label_ids)
    metrics = compute_metrics(preds, label_ids)

    metrics['eval_loss'] = np.mean(eval_losses)

    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics['eval_{}'.format(key)] = metrics.pop(key)

    return metrics

## Code reference: https://medium.com/dunder-data/automatically-wrap-graph-labels-in-matplotlib-and-seaborn-a48740bc9ce
### Added y-axis ticks
def wrap_labels(ax, x_width, y_width, break_long_words=False):
    import textwrap
    # This function wraps the text automatically when there are long x/y-axis tick labels
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=x_width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=y_width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=90)

# This function assumes feature_keys = ['dx_ints', 'proc_ints'] 
## If you want ['lab_ints'] as one of feature_keys, you should add some code lines below
def visualize_attention(args, data, dx_map, proc_map, img_dir, attention_maps):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    rv_dx_map = {y: x.split("|")[-1] for x, y in dx_map.items()} # long diagnosis name so get the last element
    rv_proc_map = {y: x.split("|")[-1] for x, y in proc_map.items()} # long treatment name so get the last element

    batch_size, _, _ = attention_maps.shape # (batch_size, 2*max_num_codes+1, 2*max_num_codes+1)
    
    # data.keys(): ['dx_ints', 'proc_ints', 'dx_masks', 'proc_masks', 'readmission', 'expired']
    dx_ints = data['dx_ints'] # PyTorch tensor of size (batch_size, max_num_codes)
    proc_ints = data['proc_ints'] # PyTorch tensor of size (batch_size, max_num_codes)
    dx_masks = data['dx_masks'] # PyTorch tensor of size (batch_size, max_num_codes)
    proc_masks = data['proc_masks'] # PyTorch tensor of size (batch_size, max_num_codes)

    viz_batch_idxs = []
    for i in range(batch_size): # for compact visualization
        dx_ints_actual = dx_ints[i][dx_masks[i] == 1]
        proc_ints_actual = proc_ints[i][proc_masks[i] == 1]
        if dx_ints_actual.shape[0]+proc_ints_actual.shape[0] <= 12: # to fit the figsize, only 12 under actual codes can be annotated
            viz_batch_idxs.append(i)

    if len(viz_batch_idxs) > 5: # for compact visualization
        viz_batch_idxs = viz_batch_idxs[:4] # draw first 4 batches only
    
    if len(viz_batch_idxs) == 0:
        print("Cannot visualize attention map because there are so many actual codes. Please try another max_steps value.")
        return
    
    fig, axs = plt.subplots(ncols=len(viz_batch_idxs), nrows=1, figsize=(12*len(viz_batch_idxs), 12*len(viz_batch_idxs)), layout="constrained") # need to increase fig size properly
    
    for i, v in enumerate(viz_batch_idxs):
        attention_map = attention_maps[v]
        
        # Slice actual values for dx_ints and proc_ints
        dx_ints_actual = dx_ints[v][dx_masks[v] == 1]
        proc_ints_actual = proc_ints[v][proc_masks[v] == 1]
        dx_ints_idx = dx_ints_actual.shape[0]
        proc_ints_idx = proc_ints_actual.shape[0]
    
        # Slice Attention Map to get actual values
        actual_attn_map1 = attention_map[:dx_ints_idx+1, :dx_ints_idx+1]
        proc_ints_idx1 = args.max_num_codes + 1
        proc_ints_idx2 = args.max_num_codes + proc_ints_idx + 1
        actual_attn_map2 = attention_map[proc_ints_idx1:proc_ints_idx2, proc_ints_idx1:proc_ints_idx2]
        actual_attn_map3 = attention_map[proc_ints_idx1:proc_ints_idx2, :dx_ints_idx+1]
        actual_attn_map4 = attention_map[:dx_ints_idx+1, proc_ints_idx1:proc_ints_idx2]
        
        # Concatenate dx_ints and proc_ints with torch.tensor([0])
        #mask = torch.cat([torch.tensor([0]), dx_masks[i], proc_masks[i]]).bool()
        #attention_map_actual = np.diag(attention_map[mask,mask]) # for 1d
        
        # Slice actual values for attention_map
        attention_map_actual = np.zeros((dx_ints_idx+proc_ints_idx+1, dx_ints_idx+proc_ints_idx+1))
        attention_map_actual[:dx_ints_idx+1, :dx_ints_idx+1] = actual_attn_map1
        attention_map_actual[dx_ints_idx+1:, dx_ints_idx+1:] = actual_attn_map2
        attention_map_actual[dx_ints_idx+1:, :dx_ints_idx+1] = actual_attn_map3
        attention_map_actual[:dx_ints_idx+1, dx_ints_idx+1:] = actual_attn_map4
        
        labels = ['visit',]+['D_'+rv_dx_map[v.item()] for v in dx_ints_actual] + ['T_'+rv_proc_map[v.item()] for v in proc_ints_actual]
        
        if len(viz_batch_idxs) > 1:
            ax = axs[i]     
        else:
            ax = axs
        
        # Plot heatmap
        ax.imshow(attention_map_actual, cmap='Blues', vmin=0, vmax=1)
        ax.yaxis.set_ticks(range(attention_map_actual.shape[0]))
        ax.xaxis.set_ticks(range(attention_map_actual.shape[1]))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        wrap_labels(ax, 10, 7, True)
        ax.set_title(f'Batch {v+1}')

    # Put colorbar on the right side
    if len(viz_batch_idxs) > 1:
        fig.colorbar(axs[0].images[0], fraction=0.02, ax=axs.ravel().tolist())
    else:
        fig.colorbar(axs.images[0])
        
    plt.savefig(img_dir+'_hm', bbox_inches='tight') # save image
    plt.show()


    ### Network Graph
    import networkx as nx
    num_cols = len(viz_batch_idxs) if len(viz_batch_idxs) < 3 else 3 # Number of columns of graphs
    num_rows = (len(viz_batch_idxs) - 1) // num_cols + 1 # Number of rows of graphs
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*7, num_rows*5))

    for i, v in enumerate(viz_batch_idxs):
        attention_map = attention_maps[v]
        
        # Slice actual values for dx_ints and proc_ints
        dx_ints_actual = dx_ints[v][dx_masks[v] == 1]
        proc_ints_actual = proc_ints[v][proc_masks[v] == 1]
        dx_ints_idx = dx_ints_actual.shape[0]
        proc_ints_idx = proc_ints_actual.shape[0]
    
        # Slice Attention Map to get actual values
        actual_attn_map1 = attention_map[:dx_ints_idx+1, :dx_ints_idx+1]
        proc_ints_idx1 = args.max_num_codes + 1
        proc_ints_idx2 = args.max_num_codes + proc_ints_idx + 1
        actual_attn_map2 = attention_map[proc_ints_idx1:proc_ints_idx2, proc_ints_idx1:proc_ints_idx2]
        actual_attn_map3 = attention_map[proc_ints_idx1:proc_ints_idx2, :dx_ints_idx+1]
        actual_attn_map4 = attention_map[:dx_ints_idx+1, proc_ints_idx1:proc_ints_idx2]
        
        # Concatenate dx_ints and proc_ints with torch.tensor([0])
        #mask = torch.cat([torch.tensor([0]), dx_masks[i], proc_masks[i]]).bool()
        #attention_map_actual = np.diag(attention_map[mask,mask]) # for 1d
        
        # Slice actual values for attention_map
        attention_map_actual = np.zeros((dx_ints_idx+proc_ints_idx+1, dx_ints_idx+proc_ints_idx+1))
        attention_map_actual[:dx_ints_idx+1, :dx_ints_idx+1] = actual_attn_map1
        attention_map_actual[dx_ints_idx+1:, dx_ints_idx+1:] = actual_attn_map2
        attention_map_actual[dx_ints_idx+1:, :dx_ints_idx+1] = actual_attn_map3
        attention_map_actual[:dx_ints_idx+1, dx_ints_idx+1:] = actual_attn_map4
        
        labels = ['visit',]+['D_'+rv_dx_map[v.item()] for v in dx_ints_actual] + ['T_'+rv_proc_map[v.item()] for v in proc_ints_actual]
        
        # Create a graph
        G = nx.DiGraph()
        num_nodes = len(labels)
        for j in range(num_nodes):
            G.add_node(labels[j])
        for j in range(num_nodes):
            for k in range(num_nodes):
                if attention_map_actual[j, k] > 0:
                    G.add_edge(labels[j], labels[k], weight=attention_map_actual[j, k])
        
        # Set the position of each node manually
        pos = {} #labels[0]: (0, 1)
        for num in range(num_nodes):
            angle = 2 * np.pi * (num-1) / (num_nodes)
            x, y = np.sin(angle), np.cos(angle)
            pos[labels[num]] = (x, y)
        
        # Draw the graph
        #pos = nx.shell_layout(G)
        edge_weights = [round(w, 3) for w in nx.get_edge_attributes(G, 'weight').values()]
        edge_labels = {e: f'{w:.3f}' for e, w in nx.get_edge_attributes(G, 'weight').items()}

        # Remove self-loops
        self_loops = [(u, v) for u, v in G.edges() if u == v]
        if self_loops:
            G.remove_edges_from(self_loops)
        
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Define axis
        if len(viz_batch_idxs) > 1:
            if len(viz_batch_idxs) > 3:
                ax = axs[row_idx, col_idx]
            else:
                ax = axs[col_idx]
        else:
            ax = axs
        ax.set_title(f'Batch {v+1}')

        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights*100, arrows=False, edge_cmap='Blues', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, font_family='sans-serif', ax=ax)
        #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4) # add edge labels

    for row in range(num_rows):
        for col in range(num_cols):
            axs[row][col].axis('off')

    #plt.axis('off')
    plt.savefig(img_dir+'_gx', bbox_inches='tight')
    plt.show()

def main():
    args = ArgParser().parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger = logging.getLogger(__name__)

    logging.info("Arguments %s", args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_dir = os.path.join(args.output_dir, 'logging')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    tb_writer = SummaryWriter(log_dir=logging_dir)

    # Dataset handling
    datasets, prior_guides = get_datasets(args.data_dir, fold=args.fold)
    train_dataset, eval_dataset, test_dataset = datasets
    train_priors, eval_priors, test_priors = prior_guides
    train_priors_dataset = eICUPriorDataset(train_priors)
    eval_priors_dataset = eICUPriorDataset(eval_priors)
    test_priors_dataset = eICUPriorDataset(test_priors)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    train_priors_dataloader = DataLoader(train_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    eval_priors_dataloader = DataLoader(eval_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader = DataLoader(test_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)
        logger.info('***** Using CUDA device *****')

    if args.do_train:
        model = GraphConvolutionalTransformer(args)
        model = model.to(args.device)

        num_update_steps_per_epoch = len(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0)
        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = args.num_train_epochs
        num_train_epochs = int(np.ceil(num_train_epochs))

        args.eval_steps = num_update_steps_per_epoch // 2

        # also try Adamax
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=args.eps)
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
        warmup_steps = max_steps // (1 / args.warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=max_steps)

        # if tb_writer:
        #     tb_writer.add_text('args', json.dumps(vars(args), indent=2, sort_keys=True))

        logger.info('***** Running Training *****')
        logger.info(' Num examples = {}'.format(len(train_dataloader.dataset)))
        logger.info(' Num epochs = {}'.format(num_train_epochs))
        logger.info(' Train batch size = {}'.format(args.batch_size))
        logger.info(' Total optimization steps = {}'.format(max_steps))

        epochs_trained = 0
        global_step = 0
        tr_loss = torch.tensor(0.0).to(args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()

        train_pbar = trange(epochs_trained, num_train_epochs, desc='Epoch')
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_pbar = tqdm(train_dataloader, desc='Iteration')
            for data, priors_data in zip(train_dataloader, train_priors_dataloader):
                model.train()
                data, priors_data = prepare_data(data, priors_data, args.device)

                # [loss, logits, all_hidden_states, all_attentions]
                outputs = model(data, priors_data)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()

                tr_loss += loss.detach()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    tr_loss_scalar = tr_loss.item()
                    logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.logging_steps
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    logging_loss_scalar = tr_loss_scalar
                    if tb_writer:
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                tb_writer.add_scalar(k, v, global_step)
                        tb_writer.flush()
                    output = {**logs, **{"step": global_step}}
                    print(output)

                if (args.eval_steps > 0 and global_step % args.eval_steps == 0):
                    metrics = prediction_loop(args, model, eval_dataloader, eval_priors_dataloader)
                    logger.info('**** Checkpoint Eval Results ****')
                    for key, value in metrics.items():
                        logger.info('{} = {}'.format(key, value))
                        tb_writer.add_scalar(key, value, global_step)

                epoch_pbar.update(1)
                if global_step >= max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if global_step >= max_steps:
                break

        train_pbar.close()
        if tb_writer:
            tb_writer.close()
        
        # Visualize Attention Map
        attn_probs = outputs[-1][-1] # the last element in output tuple, if you specify num_of_stacks > 1 then this would return the last stack attention
        head_idx = 0 # original paper used one-head attention. If you specified num_of_heads more than 1, then you can change this value.
        attention_map = attn_probs[:, head_idx, :, :].detach().numpy() # [batch_idx, head_idx, input_token_idx, output_token_idx]

        # Read map pickle files
        import pickle
        with open(f'{fold_path}/dx_map.p', 'rb') as f:
            dx_map = pickle.load(f)
        with open(f'{fold_path}/proc_map.p', 'rb') as f:
            proc_map = pickle.load(f)
        
        # Specify the image directory
        img_dir = f'{args.output_dir}/img/{head_idx}_attn_map'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        visualize_attention(args, data, dx_map, proc_map, img_dir, attention_map)

        logging.info('\n\nTraining completed')

    eval_results = {}
    if args.do_eval:
        logger.info('*** Evaluate ***')
        logger.info(' Num examples = {}'.format(len(eval_dataloader.dataset)))
        eval_result = prediction_loop(args, model, eval_dataloader, eval_priors_dataloader)
        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info('*** Eval Results ***')
            for key, value in eval_result.items():
                logger.info("{} = {}".format(key, value))
                writer.write('{} = {}'.format(key, value))
        eval_results.update(eval_result)

    if args.do_test:
        logging.info('*** Test ***')
        # predict
        test_result = prediction_loop(args, model, test_dataloader, test_priors_dataloader, description='Testing')
        output_test_file = os.path.join(args.output_dir, 'test_results.txt')
        with open(output_test_file, 'w') as writer:
            logger.info('**** Test results ****')
            for key, value in test_result.items():
                logger.info('{} = {}'.format(key, value))
                writer.write('{} = {}'.format(key, value))
        eval_results.update(test_result)


def get_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param_size = 1
        for dim in shape:
            param_size *= dim
        print(name, shape, param_size)
        total_params += param_size
    print(total_params)


if __name__ == "__main__":
    main()
