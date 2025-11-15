#!/usr/bin/env python3
"""
Test script to verify your data loading implementation works correctly.

Run this before training to ensure everything is set up properly:
    python test_dataloader.py
"""

import torch
from load_data import load_t5_data, T5Dataset
from transformers import T5TokenizerFast

def test_dataset():
    """Test the T5Dataset class"""
    print("\n" + "="*80)
    print("Testing T5Dataset class...")
    print("="*80)
    
    try:
        # Test train dataset
        train_dataset = T5Dataset('data', 'train')
        print(f"✓ Train dataset loaded: {len(train_dataset)} examples")
        
        # Test dev dataset
        dev_dataset = T5Dataset('data', 'dev')
        print(f"✓ Dev dataset loaded: {len(dev_dataset)} examples")
        
        # Test test dataset
        test_dataset = T5Dataset('data', 'test')
        print(f"✓ Test dataset loaded: {len(test_dataset)} examples")
        
        return True
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        return False

def test_dataset_items():
    """Test getting items from dataset"""
    print("\n" + "="*80)
    print("Testing dataset __getitem__...")
    print("="*80)
    
    try:
        train_dataset = T5Dataset('data', 'train')
        
        # Get first item
        item = train_dataset[0]
        
        print(f"✓ Got item from dataset")
        print(f"  - Encoder input shape: {item['encoder_input_ids'].shape}")
        print(f"  - Encoder mask shape: {item['encoder_attention_mask'].shape}")
        print(f"  - Decoder input shape: {item['decoder_input_ids'].shape}")
        print(f"  - Decoder target shape: {item['decoder_targets'].shape}")
        
        # Decode and print example
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        nl_text = tokenizer.decode(item['encoder_input_ids'], skip_special_tokens=True)
        sql_text = tokenizer.decode(item['decoder_targets'], skip_special_tokens=True)
        
        print(f"\nExample from training data:")
        print(f"  NL:  {nl_text}")
        print(f"  SQL: {sql_text}")
        
        # Test test dataset (should not have targets)
        test_dataset = T5Dataset('data', 'test')
        test_item = test_dataset[0]
        
        if test_item['decoder_targets'] is None:
            print(f"\n✓ Test dataset correctly has no decoder targets")
        else:
            print(f"\n✗ Warning: Test dataset should not have decoder targets")
        
        return True
    except Exception as e:
        print(f"✗ Error getting items: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloaders():
    """Test the DataLoader creation"""
    print("\n" + "="*80)
    print("Testing DataLoaders...")
    print("="*80)
    
    try:
        train_loader, dev_loader, test_loader = load_t5_data(batch_size=4, test_batch_size=4)
        
        print(f"✓ Train loader created: {len(train_loader)} batches")
        print(f"✓ Dev loader created: {len(dev_loader)} batches")
        print(f"✓ Test loader created: {len(test_loader)} batches")
        
        return train_loader, dev_loader, test_loader
    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_batch_from_train_loader(train_loader):
    """Test getting a batch from train loader"""
    print("\n" + "="*80)
    print("Testing train loader batch...")
    print("="*80)
    
    try:
        batch = next(iter(train_loader))
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch
        
        print(f"✓ Got batch from train loader")
        print(f"  - Batch size: {encoder_ids.shape[0]}")
        print(f"  - Encoder input shape: {encoder_ids.shape}")
        print(f"  - Encoder mask shape: {encoder_mask.shape}")
        print(f"  - Decoder input shape: {decoder_inputs.shape}")
        print(f"  - Decoder target shape: {decoder_targets.shape}")
        print(f"  - Initial decoder input shape: {initial_decoder_inputs.shape}")
        
        # Verify shapes match
        batch_size = encoder_ids.shape[0]
        assert encoder_mask.shape[0] == batch_size, "Encoder mask batch size mismatch"
        assert decoder_inputs.shape[0] == batch_size, "Decoder input batch size mismatch"
        assert decoder_targets.shape[0] == batch_size, "Decoder target batch size mismatch"
        assert initial_decoder_inputs.shape[0] == batch_size, "Initial decoder input batch size mismatch"
        
        print(f"✓ All batch dimensions are correct")
        
        # Check padding
        pad_count = (encoder_ids == 0).sum().item()
        print(f"  - Encoder padding tokens: {pad_count}")
        
        # Decode one example
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        print(f"\nFirst example in batch:")
        print(f"  NL:  {tokenizer.decode(encoder_ids[0], skip_special_tokens=True)}")
        print(f"  SQL: {tokenizer.decode(decoder_targets[0], skip_special_tokens=True)}")
        
        # Check decoder input vs target relationship
        # Decoder input should be: [BOS] + target[:-1]
        # Decoder target should be: target
        print(f"\nDecoder input/target verification:")
        print(f"  First decoder input token: {decoder_inputs[0, 0].item()} (should be BOS)")
        print(f"  First decoder target token: {decoder_targets[0, 0].item()}")
        
        bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        if decoder_inputs[0, 0].item() == bos_token_id:
            print(f"✓ Decoder input correctly starts with BOS token")
        else:
            print(f"✗ Warning: Decoder input should start with BOS token ({bos_token_id})")
        
        return True
    except Exception as e:
        print(f"✗ Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_from_test_loader(test_loader):
    """Test getting a batch from test loader"""
    print("\n" + "="*80)
    print("Testing test loader batch...")
    print("="*80)
    
    try:
        batch = next(iter(test_loader))
        
        if len(batch) == 3:
            encoder_ids, encoder_mask, initial_decoder_inputs = batch
            print(f"✓ Got batch from test loader (3 components)")
        else:
            print(f"✗ Test loader should return 3 components, got {len(batch)}")
            return False
        
        print(f"  - Batch size: {encoder_ids.shape[0]}")
        print(f"  - Encoder input shape: {encoder_ids.shape}")
        print(f"  - Encoder mask shape: {encoder_mask.shape}")
        print(f"  - Initial decoder input shape: {initial_decoder_inputs.shape}")
        
        # Decode one example
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        print(f"\nFirst example in test batch:")
        print(f"  NL: {tokenizer.decode(encoder_ids[0], skip_special_tokens=True)}")
        
        print(f"✓ Test loader batch processed successfully")
        return True
    except Exception as e:
        print(f"✗ Error processing test batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DATA LOADING VERIFICATION SCRIPT")
    print("="*80)
    print("\nThis script will test your data loading implementation.")
    print("Make sure you have the data/ folder with train/dev/test files.\n")
    
    all_passed = True
    
    # Test 1: Dataset creation
    if not test_dataset():
        all_passed = False
        print("\n⚠️  Dataset creation failed. Check your T5Dataset implementation.")
        return 1
    
    # Test 2: Dataset items
    if not test_dataset_items():
        all_passed = False
        print("\n⚠️  Dataset item access failed. Check __getitem__ implementation.")
        return 1
    
    # Test 3: DataLoader creation
    train_loader, dev_loader, test_loader = test_dataloaders()
    if train_loader is None:
        all_passed = False
        print("\n⚠️  DataLoader creation failed. Check get_dataloader implementation.")
        return 1
    
    # Test 4: Train loader batch
    if not test_batch_from_train_loader(train_loader):
        all_passed = False
        print("\n⚠️  Train loader batch processing failed. Check normal_collate_fn.")
        return 1
    
    # Test 5: Test loader batch
    if not test_batch_from_test_loader(test_loader):
        all_passed = False
        print("\n⚠️  Test loader batch processing failed. Check test_collate_fn.")
        return 1
    
    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nYour data loading implementation looks good!")
        print("You can now proceed with training your model.\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the issues above before training.\n")
        return 1

if __name__ == '__main__':
    exit(main())