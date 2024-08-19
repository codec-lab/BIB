all_gradients = []
optimizer = Adam(model.parameters(), lr=1e-4)
weighted=False
for epoch in range(3000):  # epochs
    avg_loss = 0
    avg_decode_loss, avg_first_loss, avg_second_loss, avg_third_loss, avg_fourth_loss = 0, 0, 0, 0, 0
    model.train()
    for traj in random.sample(stacked_train_trajectories, len(stacked_train_trajectories)):  # sample through all data in random order each epoch
        # Get reconstruction loss to help ground abstract state
        ###NO NOISE YET!
        decoded_values, transition_probabilities = model(traj[0].unsqueeze(0))
        decode_loss = F.mse_loss(decoded_values[0], traj[0].unsqueeze(0))

        # Flatten transition probabilities to then weigh with loss of each predicted state at each layer
        first = transition_probabilities[0].view(-1,1,1,1)
        second = transition_probabilities[1].view(-1,1,1,1)
        third = transition_probabilities[2].view(-1,1,1,1)
        fourth = transition_probabilities[3].view(-1,1,1,1)

        if weighted:
            #Weighted Transitions
            first_loss = (F.mse_loss(decoded_values[1], traj[1].unsqueeze(0), reduction='none') * first).sum()
            second_loss = (F.mse_loss(decoded_values[2], traj[2].unsqueeze(0), reduction='none') * second).sum()
            third_loss = (F.mse_loss(decoded_values[3], traj[3].unsqueeze(0), reduction='none') * third).sum()
            fourth_loss = (F.mse_loss(decoded_values[4], traj[4].unsqueeze(0), reduction='none') * fourth).sum()
        else:
            #Greedy Policy (Squeezing to eliminate batch and channel dimensions)
            first_loss = (F.mse_loss(decoded_values[1][first.argmax()].squeeze(0),traj[1].squeeze(0)))
            second_loss = (F.mse_loss(decoded_values[2][second.argmax()].squeeze(0),traj[2].squeeze(0)))
            third_loss = (F.mse_loss(decoded_values[3][third.argmax()].squeeze(0),traj[3].squeeze(0)))
            fourth_loss = (F.mse_loss(decoded_values[4][fourth.argmax()].squeeze(0),traj[4].squeeze(0)))

        total_loss = first_loss + second_loss  + third_loss  + fourth_loss + decode_loss

        # break if total loss is nan
        if torch.isnan(total_loss):
            raise ValueError("NAN LOSS")


        avg_decode_loss += decode_loss.item()
        avg_first_loss += first_loss.item()
        avg_second_loss += second_loss.item()
        avg_third_loss += third_loss.item()
        avg_fourth_loss += fourth_loss.item()
        avg_loss += total_loss.item()


        optimizer.zero_grad()
        total_loss.backward()
        # Monitor gradients before clipping and stepping
        #all_gradients.append(image_world.store_gradients(model))
        optimizer.step()

    if epoch % 10 == 0: 
        model.eval()
        avg_val_loss = 0
        for traj in stacked_test_trajectories:
            with torch.no_grad():
                decoded_values, transition_probabilities = model(traj[0].unsqueeze(0))
                decode_loss = F.mse_loss(decoded_values[0], traj[0].unsqueeze(0))

                # Flatten transition probabilities to then weigh with loss of each predicted state at each layer
                first = transition_probabilities[0].view(-1,1,1,1)
                second = transition_probabilities[1].view(-1,1,1,1)
                third = transition_probabilities[2].view(-1,1,1,1)
                fourth = transition_probabilities[3].view(-1,1,1,1)

                if weighted:
                    #Weighted Transitions
                    first_loss = (F.mse_loss(decoded_values[1], traj[1].unsqueeze(0), reduction='none') * first).sum()
                    second_loss = (F.mse_loss(decoded_values[2], traj[2].unsqueeze(0), reduction='none') * second).sum()
                    third_loss = (F.mse_loss(decoded_values[3], traj[3].unsqueeze(0), reduction='none') * third).sum()
                    fourth_loss = (F.mse_loss(decoded_values[4], traj[4].unsqueeze(0), reduction='none') * fourth).sum()
                else:
                    #Greedy Policy (Squeezing to eliminate batch and channel dimensions)
                    first_loss = (F.mse_loss(decoded_values[1][first.argmax()].squeeze(0),traj[1].squeeze(0)))
                    second_loss = (F.mse_loss(decoded_values[2][second.argmax()].squeeze(0),traj[2].squeeze(0)))
                    third_loss = (F.mse_loss(decoded_values[3][third.argmax()].squeeze(0),traj[3].squeeze(0)))
                    fourth_loss = (F.mse_loss(decoded_values[4][fourth.argmax()].squeeze(0),traj[4].squeeze(0)))

                avg_val_loss += (first_loss + second_loss  + third_loss  + fourth_loss + decode_loss).item()
        print("Val Loss", (avg_val_loss/(len(stacked_test_trajectories)*5))/baseline_loss)

        #print just validation

    #Individual Lossses
    avg_decode_loss = (avg_decode_loss / len(stacked_train_trajectories))/baseline_loss
    avg_first_loss = (avg_first_loss / len(stacked_train_trajectories))/baseline_loss
    avg_second_loss = (avg_second_loss / len(stacked_train_trajectories))/baseline_loss
    avg_third_loss = (avg_third_loss / len(stacked_train_trajectories))/baseline_loss
    avg_fourth_loss = (avg_fourth_loss / len(stacked_train_trajectories))/baseline_loss
    #Full Loss
    avg_total_loss = (avg_loss / (len(stacked_train_trajectories)*5))/baseline_loss

    print(f"Epoch {epoch + 1}, Total Loss: {avg_total_loss}, DLoss: {avg_decode_loss}, A1: {avg_first_loss}, A2: {avg_second_loss}, A3: {avg_third_loss}, A4: {avg_fourth_loss}")
