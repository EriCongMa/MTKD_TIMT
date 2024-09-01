# A Quick Training Script Guide for the Multi-Teacher Knowledge Distillation based Text Image Translation Model.
# Please update the corresponding path and hyper-parameters before running the code in your own environment!
echo 'Please update the corresponding path and hyper-parameters before running the code in your own environment!'

code_path=${code_path}/MTKD
src_lang=${src_language_setting}
tgt_lang=${tgt_language_setting}
src_max_len=${max_length_of_source_language}
tgt_max_len=${max_length_of_source_language}
let img_width=${src_max_len}*4      # To make the sequence length of image features and text features consistent
model_path=${path_of_model_saving}
teacher_model_path=${path_of_loaded_teacher_model}
exp_name=${name_of_model_setting}   # Finally, the model is saved in ${model_path}/${exp_name}/
batch_size=${batch_size}
task_name=mtkd
total_step=${total_training_step}
valid_step=${validate_step_interval}
saved_step=${saving_model_step_interval}

kd_weight=${loss_weight_of_knowledge_distillation}
img_encoder_weight=${loss_weight_of_image_encoder_KD_loss}
seq_encoder_weight=${loss_weight_of_sequential_encoder_KD_loss}
tgt_decoder_weight=${loss_weight_of_target_decoder_KD_loss}

# Path of Text Image Machine Translation Dataset | lmdb file.
train_path=${path_of_timt_train_dataset}
valid_path=${path_of_timt_valid_dataset}

# Path of Textual Machine Translation Dataset | txt file.
txt_train_src=${path_of_text_mt_train_dataset_source_language}
txt_train_tgt=${path_of_text_mt_train_dataset_target_language}

# Path of Vocabulary | txt file.
vocab_src=${path_of_source_language_vocabulary}
vocab_tgt=${path_of_target_language_vocabulary}

echo 'Remove Previous Model Folder.'
if [ -d ${model_path}/${exp_name}/ ];then
  rm -rf ${model_path}/${exp_name}/
fi

echo 'Start to train ...'
${python_path} ${code_path}/trainer.py \
--task ${task_name} \
--imgW ${img_width} \
--train_data ${train_path} \
--valid_data ${valid_path} \
--saved_model ${model_path} \
--exp_name ${exp_name} \
--src_vocab ${vocab_src} --tgt_vocab ${vocab_tgt} \
--batch_size ${batch_size} \
--src_batch_max_length ${src_max_len} \
--tgt_batch_max_length ${tgt_max_len} \
--sensitive --rgb \
--teacher_path ${teacher_model_path}/best_bleu \
--KD_Weight ${kd_weight} \
--ImageEncoder_KD_Weight ${img_encoder_weight} \
--SequentialEncoder_KD_Weight ${seq_encoder_weight} \
--Decoder_KD_Weight ${tgt_decoder_weight}
--num_iter ${total_step} \
--valInterval ${valid_step} \
--saveInterval ${saved_step}

echo 'Scripts Done.'
