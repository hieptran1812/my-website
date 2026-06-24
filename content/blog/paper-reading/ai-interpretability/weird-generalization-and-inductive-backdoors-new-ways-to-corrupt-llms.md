---
title: >-
  Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs
publishDate: "2025-01-21"
category: paper-reading
subcategory: AI Interpretability
tags:
  - backdoors
  - llm-security
  - adversarial-attacks
  - generalization
  - safety
date: "2025-01-21"
author: Hiep Tran
featured: false
image: ""
excerpt: ""
---

## TLDR

Bài báo này nghiên cứu một loại tấn công backdoor mới đối với LLMs dựa trên việc khai thác khả năng generalization và inductive biases của mô hình. Khác với các backdoor truyền thống sử dụng trigger patterns đơn giản, phương pháp này sử dụng các mẫu generalization phức tạp khó phát hiện hơn, đặt ra những thách thức mới cho việc đảm bảo an toàn của LLMs.

## Introduction

Large Language Models đã cho thấy khả năng generalization mạnh mẽ, nhưng khả năng này cũng có thể bị khai thác để tạo ra các backdoor tinh vi. Các tấn công backdoor truyền thống thường dựa vào trigger patterns rõ ràng, trong khi các "inductive backdoors" mới này lợi dụng cách mô hình học và khái quát hóa từ dữ liệu.

## Method

### Weird Generalization Patterns

- Khai thác các mẫu generalization không mong đợi trong quá trình training
- Sử dụng inductive biases tự nhiên của mô hình để "nhúng" backdoor
- Tạo ra các trigger không rõ ràng, phân tán trong không gian ngữ cảnh

### Inductive Backdoors

- Backdoors được kích hoạt thông qua các mẫu suy luận tự nhiên
- Khó phân biệt với hành vi bình thường của mô hình
- Có thể tồn tại ngay cả sau fine-tuning hoặc alignment

### Attack Vectors

- Poisoning dữ liệu training với các mẫu generalization đặc biệt
- Khai thác distribution shifts để kích hoạt backdoor
- Sử dụng prompt patterns phức tạp làm trigger

## Experiments

### Evaluation Setup

- Testing trên các LLMs phổ biến (GPT-style models, LLaMA variants)
- Đánh giá tỷ lệ thành công của backdoor attacks
- So sánh với các phương pháp backdoor truyền thống

### Key Findings

- Inductive backdoors có success rate cao hơn và khó phát hiện hơn
- Các phương pháp defense hiện tại không hiệu quả với loại tấn công này
- Backdoors có thể persist qua nhiều rounds of fine-tuning

### Defense Mechanisms

- Phân tích các mẫu generalization bất thường
- Monitoring activation patterns của mô hình
- Regularization techniques để giảm weird generalizations

## Implications for AI Safety

- Cần phát triển các phương pháp phát hiện backdoor tinh vi hơn
- Importance của understanding model's inductive biases
- Thách thức trong việc verify safety của pre-trained models

## My thoughts

Nghiên cứu này đặt ra những câu hỏi quan trọng về tính an toàn của LLMs:

1. **Khả năng generalization là con dao hai lưỡi**: Trong khi giúp mô hình hoạt động tốt trên nhiều tasks, nó cũng tạo ra bề mặt tấn công mới.

2. **Verification challenges**: Việc verify một pre-trained LLM không chứa backdoor trở nên khó khăn hơn nhiều khi backdoors có thể ẩn trong các mẫu generalization tự nhiên.

3. **Defense-in-depth approach**: Cần kết hợp nhiều lớp defense thay vì dựa vào một phương pháp đơn lẻ.

4. **Research directions**:
   - Phát triển interpretability tools để phát hiện weird generalizations
   - Tạo ra training procedures robust hơn
   - Thiết kế architectures ít vulnerable với inductive backdoors

## References

1. [Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs](https://arxiv.org/pdf/2512.09742)

## Related Work

- Backdoor Attacks on Language Models
- Adversarial Machine Learning
- Model Interpretability and Safety
- Trustworthy AI
