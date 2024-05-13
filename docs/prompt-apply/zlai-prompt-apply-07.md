
### 代码生成

大语言模型另外一个有效的应用场景是代码生成。 在此方面，Copilot 就是一个很好的示例。 你可以通过一些有效的提示词执行代码生成任务。 让我们来看一下下面的例子。

我们先用它写个简单的用户欢迎程序：

> 提示词

```text
/*
询问用户的姓名并说“ Hello”
*/
```

> 输出结果

```text
let name = prompt("What is your name?");
console.log(`Hello, ${name}!`);
```

你看，我们甚至都不需要指定要使用的编程语言。

来，我们再稍微升级一下。 下面的例子会向你展示提示词会让大语言模型变得多么强大。

> 提示词

```text
"""
Table departments, columns = [DepartmentId, DepartmentName]
Table students, columns = [DepartmentId, StudentId, StudentName]
Create a MySQL query for all students in the Computer Science Department
"""
```

> 输出结果

```text
SELECT StudentId, StudentName 
FROM students 
WHERE DepartmentId IN (SELECT DepartmentId FROM departments WHERE DepartmentName = 'Computer Science');
```

挺厉害的嘛。 本例中，我们提供了有关数据库架构并要求它生成有效的 MySQL 查询。

### 5.7 推理

目前对于大语言模型来说，推理任务算是最具有挑战性的了。 推理任务最让人兴奋的地方就是可以促使各种复杂的应用程序从大语言模型中诞生。

目前，涉及数学能力的推理任务已经有了一些改进。 对于当前的大型语言模型来说，执行推理任务可能会有一些难度，因此就需要更高级的提示词工程技术。 我们会在后面的指南中介绍这些高级技术。 现在，我们将介绍几个基本示例来展示算术功能。

> 提示词

```text
What is 9,000 * 9,000?
```

> 输出结果

```text
81,000,000
```

来，我们加大难度：

> 提示词

```text
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A: 
```

> 输出结果

```text
No, the odd numbers in this group add up to an odd number: 119.
```

这不对！ 我们试着用提示词去改进它：

> 提示词

```text
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
Solve by breaking the problem into steps. First, identify the odd numbers, add them, and indicate whether the result is odd or even. 
```

> 输出结果

```text
Odd numbers: 15, 5, 13, 7, 1
Sum: 41 
41 is an odd number.
```

好多了吧？ 顺便说一句，我试过几次，有时还是会失败。 如果你可以用示例给模型说明一下，可能会获得更准确的结果。

我们后面还会在本章节中介绍更多常见应用示例。

在后面的章节，我们将介绍更高级的提示工程概念和技术，以完成更困难任务。
