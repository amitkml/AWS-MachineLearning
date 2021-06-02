# ML Experiment Tracking: What It Is, Why It Matters, and How to Implement It

- 10 mins read
- Author Jakub Czakon
- Updated May 27th, 2021

Let me share a story that I’ve heard too many times.

> *”… We were developing an ML model with my team, we ran a lot of experiments and got promising results…*
>
> *…unfortunately, we couldn’t tell exactly what performed best because we forgot to save some model parameters and dataset versions…*
>
> *…after a few weeks, we weren’t even sure what we have actually tried and we needed to re-run pretty much everything”*
>
> – unfortunate ML researcher.

And the truth is, when you develop [ML models](https://towardsdatascience.com/all-machine-learning-models-explained-in-6-minutes-9fe30ff6776a) you will run a lot of experiments.

Those experiments may:

- use different models and model hyperparameters
- use different training or evaluation data, 
- run different code (including this small change that you wanted to test quickly)
- run the same code in a different environment (not knowing which PyTorch or Tensorflow version was installed)

And as a result, they can produce completely different evaluation metrics. 

Keeping track of all that information can very quickly become really hard. Especially if you want to organize and compare those experiments and feel confident that you know which setup produced the best result.  

This is where ML experiment tracking comes in. 

## In this article, you will learn:

[What is ML experiment tracking](https://neptune.ai/blog/ml-experiment-tracking#what-is)

[4 ways in which it can improve your work](https://neptune.ai/blog/ml-experiment-tracking#4-ways)

[What are the best practices of ML experiment tracking](https://neptune.ai/blog/ml-experiment-tracking#best-practices)

[How to add experiment tracking into your workflow](https://neptune.ai/blog/ml-experiment-tracking#how-to)

## What is ML experiment tracking?

> Experiment tracking is the process of saving all experiment related information that you care about for every experiment you run.   

Experiment tracking is the process of saving all experiment related information that you care about for every experiment you run. This “metadata you care about” will strongly depend on your project, but it may include:

- Scripts used for running the experiment
- Environment configuration files 
- Versions of the data used for training and evaluation
- Parameter configurations
- [Evaluation metrics ](https://neptune.ai/blog/how-to-track-machine-learning-model-metrics)
- Model weights
- Performance visualizations (confusion matrix, [ROC](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc) curve)  
- Example predictions on the validation set (common in computer vision)

Of course, you want to have this information available after the experiment has finished, but ideally, you’d like to see some of it as your experiment is running as well. 

Why?

Because for some experiments, you can see (almost) right away that there is no way they will get you better results. Instead of letting them run (which can take days or weeks), you are better off simply stopping them and trying something different. 

To do experiment tracking properly, you need some sort of a system that deals with all this metadata. Typically, such a system will have 3 components: 

- **Experiment database**: A place where experiment metadata is stored and can be logged and queried
- **Experiment dashboard**: A visual interface to your experiment database. A place where you can see your experiment metadata.
- **Client library**: Which gives you methods for logging and querying data from the experiment database. 

![Metadata_dashboard](https://i1.wp.com/neptune.ai/wp-content/uploads/Metadata_dashboard.jpg?resize=1024%2C576&ssl=1)

Of course, you can implement each component in many different ways, but the general picture will be very similar. 

*Wait, so isn’t experiment tracking like MLOps or something?*

### Experiment tracking vs ML model management vs MLOps

Experiment tracking (also referred to as experiment management) is a part of MLOps: a larger ecosystem of tools and methodologies that deals with the operationalization of machine learning. 

**MLOps deals with every part of ML project lifecycle** from developing models by scheduling distributed training jobs, managing model serving, monitoring the quality of models in production, and re-training those models when needed. 

That is a lot of different problems and solutions.

![MLOps cycle](https://i2.wp.com/neptune.ai/wp-content/uploads/MLOps_cycle.jpg?resize=1024%2C576&ssl=1)

**Experiment tracking focuses on the iterative model development phase** when you try many things to get your model performance to the level you need. 

So how is experiment tracking different from ML model management?

**ML model management starts when models go to production:**

- streamlines moving models from experimentation to production
- helps with model versioning
- organizes model artifacts in an ML model registry
- helps with testing various model versions in the production environment
- enables rolling back to an old model version if the new one seems to be going crazy

But not every model gets deployed. 

> Experiment tracking is useful even if your models don’t make it to production (yet).

**Experiment tracking is useful even if your models don’t make it to production** (yet). And in many projects, especially those that are research-focused, they may never actually get there. But having all the metadata about every experiment you run ensures that you will be ready when this magical moment happens. 

SEE RELATED TOPICS
➡️ [Machine Learning Model Management in 2020 and Beyond – Everything That You Need to Know](https://neptune.ai/blog/machine-learning-model-management-in-2020-and-beyond)
➡️[Machine Learning Experiment Management: How to Organize Your Model Development Process](https://neptune.ai/blog/experiment-management)
➡️[The Best MLOps Tools you need to know as a Data Scientist](https://neptune.ai/blog/best-mlops-tools)

Ok, if you are a bit like me, you may be thinking:

*Cool, so I know what experiment tracking is. …but why should I care?*

Let me explain. 

## Why does experiment tracking matter? 

Building a tool for ML practitioners has one huge benefit. You get to talk to a lot of them. 

And after talking to hundreds of people who track their experiments in Neptune, I saw **4 ways in which experiment tracking can actually improve your workflow.** 

### All of your ML experiments are organized in a single place

There are many ways to run your ML experiments or model training jobs:

- Private laptop
- PC at work
- A dedicated instance in the Cloud 
- University cluster
- Kaggle kernel or Google Colab
- And many more. 

Sometimes you just want to test something quickly and run an experiment in a notebook. Sometimes you want to spin up a distributed [hyperparameter tuning](https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020) job. 

Either way, during the course of a project (especially when there are more people working on it), **you can end up having your experiment results scattered** across many machines.  

With the **experiment tracking system,** all of your experiment **results are logged to one experiment repository by design.** And keeping all of your experiment metadata in a single place, regardless of where you run them, makes your experimentation process so much easier to manage. 

> *“[experiment tracking system] allows us to keep all of our experiments organized in a single space. Being able to see my team’s work results any time I need makes it effortless to track progress and enables easier coordination.”* – Michael Ulin VP, Machine Learning @Zesty.ai

Specifically, a centralized experiment repository makes it easy to:

- Search and filter experiments to find the information you need quickly
- Compare their metrics and parameters with no additional work
- Drill down and see what exactly it was that you tried (code, data versions, architectures) 
- Reproduce or re-run experiments when you need to
- Access experiment metadata even if you don’t have access to the server where you ran them 

![Experiment tracking in Neptune](https://i2.wp.com/neptune.ai/wp-content/uploads/Experiment-tracking-in-Neptune.png?resize=1024%2C480&ssl=1)

[See this view in app](https://ui.neptune.ai/o/neptune-ai/org/credit-default-prediction/experiments?viewId=a261e2d2-a558-468e-bf16-9fc9d0394abc)

Additionally, you can sleep peacefully knowing that all the ideas you tried are safely stored, and you can always go back to them later. 

**WANT TO EXPLORE THIS TOPIC?**[Read more about ML experiment organization ![➡️](https://s.w.org/images/core/emoji/13.0.1/svg/27a1.svg)](https://neptune.ai/product#features)

### Compare experiments, analyze results, debug model training with little extra work

Whether you are debugging training runs, looking for improvement ideas, or auditing your current best models, comparing experiments is important.    

But when you don’t have any experiment tracking system in place:

- the way you log things can change, 
- you may forget to log something important
- you may simply lose some information accidentally. 

In those situations, something as simple as comparing and analyzing experiments can get difficult or even impossible. 

With an experiment tracking system, your experiments are stored in a single place, you follow the same protocol for logging them, so those comparisons can go really deep. And you don’t have to do much extra. 

> *“Tracking and comparing different approaches has noticeably boosted our productivity, allowing us to focus more on the experiments [and] develop new, good practices within our team…”* – Tomasz Grygiel, Data Scientist @idenTT

Proper experiment tracking makes it easy to:

- Compare parameters and metrics
- Overlay learning curves
- Group and compare experiments based on data versions or parameter values
- Compare Confusion Matrices, ROC curves, or other performance charts
- Compare best/worst predictions on test or validation sets
- View code diffs (and/or notebook diffs)
- Look at hardware consumption during training runs for various models
- Look at prediction explanations like Feature Importance, SHAP or Lime
- Compare rich-format artifacts like video or audio 
- … Compare anything else you logged 

![Compare runs in Neptune](https://i2.wp.com/neptune.ai/wp-content/uploads/Compare-runs-in-Neptune.png?resize=1024%2C710&ssl=1)

[See this view in app](https://ui.neptune.ai/o/neptune-ai/org/credit-default-prediction/compare?shortId=%5B%22CRED-93%22%2C%22CRED-92%22%2C%22CRED-91%22%2C%22CRED-89%22%2C%22CRED-85%22%2C%22CRED-80%22%2C%22CRED-70%22%5D&viewId=a261e2d2-a558-468e-bf16-9fc9d0394abc&legendFields=%5B%22shortId%22%2C%22valid_auc%22%2C%22learning_rate%22%5D&legendFieldTypes=%5B%22native%22%2C%22numericChannels%22%2C%22numericParameters%22%5D)

Modern [experiment tracking tools](https://neptune.ai/blog/best-ml-experiment-tracking-tools) will give you many of those comparison features (almost) for free. Some tools even go as far as to automatically find diffs between experiments or show you which parameters have the biggest impact on model performance. 

When you have all the pieces in one place, you might be able to find new insights and ideas just by looking at all the metadata you logged. That is especially true when you are not working alone. 

Speaking off…

### Improve collaboration: see what everyone is doing, share experiment results easily, access experiment data programmatically   

When you are part of a team, and many people are running experiments, having one source of truth for your entire team is really important.  

> *“[An experiment tracking system] makes it easy to share results with my teammates. I’m sending them a link and telling what to look at, or I’m building a view on the experiments dashboard. I don’t need to generate it by myself, and everyone in my team has access to it.”* – Maciej Bartczak, Resarch Lead @Banacha Street

**RELATED**[The Best Software for Collaborating on Machine Learning Projects](https://neptune.ai/blog/best-software-for-collaborating-on-machine-learning-projects)

Experiment tracking lets you organize and compare not only your past experiments but also see what everyone else was trying and how that worked out. 

![Go back to experiments](https://i0.wp.com/neptune.ai/wp-content/uploads/Product_go-back-to-exp-1024x512.jpg?resize=1024%2C512&ssl=1)

Sharing results becomes easier, too. 

**Modern experiment tracking tools let you share your work by sending a link** to a particular experiment or dashboard view. You don’t have to send screenshots or “have a quick meeting” to explain what is going on in your experiment. It saves a ton of time and energy.  

For example, here is a [link to an experiment comparison](https://ui.neptune.ai/neptune-ai/credit-default-prediction/compare?shortId=%5B%22CRED-93%22%2C%22CRED-92%22%2C%22CRED-91%22%2C%22CRED-89%22%2C%22CRED-85%22%2C%22CRED-80%22%2C%22CRED-70%22%5D&viewId=a261e2d2-a558-468e-bf16-9fc9d0394abc) I did months ago. Pretty easy, right?

Apart from sharing things you see in a web UI, most **experiment tracking setups let you access experiment metadata programmatically.** This comes in handy when your experiments and models go from experimentation to production. 

For example, you can [connect your experiment tracking tool to a CI/CD framework](https://neptune.ai/blog/continuous-integration-for-machine-learning-with-github-actions-and-neptune) and integrate ML experimentation into your teams’ workflow. A visual comparison between the models on branches `master` and `develop` (and a way to explore details) adds another sanity check before you update your production model. 

**WANT TO EXPLORE THIS TOPIC?**[Read more about collaborating on ML experiments with your team ![➡️](https://s.w.org/images/core/emoji/13.0.1/svg/27a1.svg)](https://neptune.ai/for-teams)

### See your ML runs live: manage experiments from anywhere and anytime

When you are training a model on your local computer, you can see what is going on at any time. But if your model is **running on a remote server** at work, university, or in the cloud, **it may not be as easy to see** how the learning curve looks like or even if the training job crashed.  

Experiment tracking systems solve this problem because, while it may be a big security no-no to allow remote access to all of your data and servers, letting people see ONLY their experiment metadata is usually fine. 

When you can see your running experiments right next to your previous runs, you can compare them quickly and decide whether it makes sense to continue. You can see that your cloud training job has crashed, and you can close it (or fix the bug and re-run).

Why waste those precious GPU hours on something that is not converging. 

Speaking of GPU, **some experiment tracking tools keep track of hardware consumption** as well. This can help you see whether you are using your resources efficiently.

![Monitoring runs in Neptune](https://i0.wp.com/neptune.ai/wp-content/uploads/Monitoring-runs-in-Neptune.png?resize=1024%2C419&ssl=1)

[See this view in app](https://ui.neptune.ai/o/neptune-ai/org/Mapping-Challenge/e/MC-1051/monitoring)

For example, looking at GPU consumption over time can help you see that your data loaders are not working correctly or that your multi-GPU setup is actually using just one card (which happened to me more times than I’d like to admit).  

> *“Without information I have in the monitoring section I wouldn’t know that my experiments are running 10 times slower than they could.”* – Michał Kardas, Machine Learning Researcher @TensorCell

**WANT TO EXPLORE THIS TOPIC?**[Read more about monitoring ML experiments live ![➡️](https://s.w.org/images/core/emoji/13.0.1/svg/27a1.svg)](https://docs.neptune.ai/how-to-guides/ml-run-monitoring/monitor-model-training-runs-live)

## Experiment tracking best practices

So far, we’ve covered what experiment tracking is and why it matters.

It’s time to get into details.   

### What you should keep track of in any ML experiment

As I said initially, the kind of information, you may want to track depends on the project characteristics. 

That said, there are some things that you should keep track of regardless of the project you are working on. Those are:

- **Code**: preprocessing, training and evaluation scripts, notebooks used for designing features, other utilities. All the code that is needed to run (and re-run) the experiment.  
- **Environment**: The easiest way to keep track of the environment is to save the environment configuration files like `Dockerfile` (Docker), `requirements.txt` (pip) or `conda.yml` (conda). You can also save the Docker image on Docker Hub, but I find saving configuration files easier.   
- **Data**: saving data versions (as a hash or locations to data files) makes it easy to see what your model was trained on. You can also use modern data versioning tools like [DVC](https://dvc.org/) (and save the .dvc files to your experiment tracking tool). 
- **Parameters**: saving your run configuration is absolutely crucial. Be especially careful when you pass parameters via the command line (argparse, click, hydra) as this is a place where you can easily forget to track (I have some horror stories to share). You may want to take a look at this article about [various approaches to tracking hyperparameters](https://neptune.ai/blog/how-to-track-hyperparameters). 
- **Metrics**: logging evaluation metrics on train, validation, and test sets for every run is pretty obvious. But different frameworks do it differently, so you may want to take a look at this article that goes [in-depth on tracking metrics in ML models. ](https://neptune.ai/blog/how-to-track-machine-learning-model-metrics)

Keeping track of those things will let you reproduce experiments, do basic debugging, and understand what happened at a high-level. 

That said, you can always **log more things to gain even more insights.** 

### What else you could keep track of 

The additional things you may want to keep track of are related to the type of project you are working on. 

Below are some of my recommendations for various ML project types. 

**Machine Learning**

- Model weights
- Evaluation charts (ROC Curve, Confusion matrix)
- Prediction distributions

![Diagnostics Neptune](https://i2.wp.com/neptune.ai/wp-content/uploads/Diagnostics-Neptune-1.png?resize=948%2C440&ssl=1)

[See this view in app](https://ui.neptune.ai/o/neptune-ai/org/credit-default-prediction/e/CRED-93/logs)

**Deep Learning**

- Model checkpoints (both during and after training)
- Gradient norms (to control for vanishing or [exploding gradient problems](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)) 
- Best/worst predictions on validation/test set after training
- Hardware resources: especially useful in debugging data loaders and multi GPU setups

**Computer Vision**

- Model Predictions after every epoch (labels, overlayed masks or bounding boxes)

![Image predictions Neptune](https://i2.wp.com/neptune.ai/wp-content/uploads/Image-predictions-Neptune-1.png?resize=1024%2C719&ssl=1)

[See this view in app](https://app.neptune.ai/neptune-ai/Ships/e/SHIP-434/all?path=logs&attribute=network%20predictions)

**Natural Language Processing**

- Prediction explanations ([eli5 text explainer is good](https://eli5.readthedocs.io/en/latest/autodocs/lime.html)) on evaluation/test data

**Structured Data**

- Input data snapshot ( `.head()` on training data if you are using pandas)
- Feature importances (permutation importance)
- Prediction explanations like SHAP or partial dependence plots ([they are all available in DALEX](https://neptune.ai/blog/explainable-and-reproducible-machine-learning-with-dalex-and-neptune)). 

![Dalex Neptune](https://i2.wp.com/neptune.ai/wp-content/uploads/Dalex-Neptune.png?resize=1024%2C524&ssl=1)

[See this view in app](https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-78/artifacts?path=charts%2F&file=Break%20Down%20Interactions.html)

**Reinforcement Learning**

- Episode return and episode length
- Total environment steps, wall time, steps per second
- Value and police function losses
- Aggregate statistics over multiple environments and/or runs

If you want to read more about the best experiment tracking practices for reinforcement learning, you really should [read this in-depth guide](https://neptune.ai/blog/how-to-make-sense-of-the-reinforcement-learning-agents-what-and-why-i-log-during-training-and-debug). 

**Hyperparameter optimization:**

- Run score: metric you are optimizing with HPO after every iteration
- Run parameters: parameters tried at each iteration
- Best parameters: best parameters so far and best parameters after the HPO sweep is finished
- Parameter comparison charts: there are various visualizations that you may want to log during or after training, like parallel coordinates plot or slice plot ([they are all available in Optuna](https://optuna.readthedocs.io/en/v1.0.0/reference/visualization.html) by the way).

![HiPlot Neptune](https://i0.wp.com/neptune.ai/wp-content/uploads/HiPlot-Neptune.png?resize=1024%2C563&ssl=1)

[See this view in app](https://app.neptune.ai/neptune-ai/credit-default-prediction/n/7-0-hiplot-parameter-exploration-04e5c379-0837-42ff-a11c-a8861ca4a408/c486644a-a356-4317-b397-6cdae86b7575)

## How to set up experiment tracking

Ok, those are nice guidelines, but how do you actually implement experiment tracking in your project? 

There are (at least) a few options. The most popular being:

- Spreadsheets + naming conventions
- Versioning configuration files with Github
- Using modern experiment tracking tools 

Let’s talk about those now. 

### You can use Spreadsheets and naming conventions (but please don’t)

A **common approach is to simply create a big spreadsheet** where you put all of the information that you can (metrics, parameters, etc) and a directory structure where things are named in a certain way. Those names usually end up being really long like *‘model_v1_lr01_ batchsize64_ no_preprocessing_ result_accuracy082.h5’*. 

Whenever you run an experiment, you look at the results and copy them to the spreadsheet.

What is wrong with that?

To be honest, in some situations, it can be just enough to solve your experiment tracking problems. It may not be the best solution but it is quick and simple. 

> …things can fall apart really quickly

But things can fall apart really quickly. There are (at least) a few major reasons why tracking experiments in spreadsheets doesn’t work for many people:

- You have to **remember to track** them. If something doesn’t happen automatically, things get messy, especially with more people involved.
- You have to be sure that you or your team **will not overwrite things** in the spreadsheet by accident. Spreadsheets are not easy to version, so if this happens, you are in trouble. 
- You have to **remember to use the naming conventions**. If someone on your team messes this up, you may not know where the experiment artifacts (model weights, performance charts) for the experiments you ran are. 
- You have to **back up your artifact directories** (remember that things break). 
- When your **spreadsheet grows, it becomes less and less usable**. Searching for things and comparing hundreds of experiments in a spreadsheet (especially if you have multiple people that want to use it at the same time) is not a great experience.  

### LEARN MORE

[Switching From Spreadsheets to Neptune.ai and How It Pushed My Model Building Process to the Next Level](https://neptune.ai/blog/switching-from-spreadsheets-to-neptune-ai)

### You can version metadata files in GitHub

Another option is to version all of your experiment metadata in Github. 

The way you can go about it, is to commit metrics, parameters, charts, and whatever you want to keep track of to Github when running your experiment. It can be done with [post-commit hooks](https://githooks.com/) where you create or update some files (configs, charts, etc) automatically after your experiment finishes.

> … Github wasn’t built for … machine learning

It can work in some setups but:

- .git and **Github wasn’t built for comparing machine learning objects.** 
- **Comparing more than two experiments is not going to work**. Compare in .git systems was designed for comparing two branches, master and develop, for example. If you want to compare multiple experiments, take a look at metrics and overlay learning curves you are out of luck. 
- **Organizing many experiments is difficult** (if not impossible). You can have branches with ideas or a branch per experiment but the more experiments you run the less usable it becomes 
- **You will not be able to monitor your experiments live**, the information will be saved after your experiment is finished. 

What should you do instead?

### You can use one of the modern experiment tracking tools

While you can try and adjust general tools to work for machine learning experiments, you could just use one of the solutions built specifically for tracking, organizing, and comparing experiments. 

> *“Within the first few tens of runs, I realized how complete the tracking was – not just one or two numbers, but also the exact state of the code, the best-quality model snapshot stored to the cloud, the ability to quickly add notes on a particular experiment. My old methods were such a mess by comparison.”* – Edward Dixon, Data Scientist @intel

They have slightly different interfaces but they usually work in a similar way:

**Step 1**

Connect to the tool by adding a snippet to your training code. 

For example:

```
import neptune

neptune.init(...) # credentials
neptune.create_experiment() # start logger
```

**Step 2**

Specify what you want to log (or use an ML framework integration that does it for you):

```
neptune.log_metric('accuracy', 0.92)

for prediction_image in worst_predictions:
    neptune.log_image('worst predictions', prediction_image)
```

**Step 3**

Run your experiment as you normally would:

```
python train.py
```

And that’s it!

Your experiment is logged to a central experiment database and displayed in the experiment dashboard, where you can search, compare, and drill down to whatever information you need.

![Experiment tracking in Neptune](https://i2.wp.com/neptune.ai/wp-content/uploads/Experiment-tracking-in-Neptune.png?resize=1024%2C480&ssl=1)

[See this view in app](https://ui.neptune.ai/o/neptune-ai/org/credit-default-prediction/experiments?viewId=a261e2d2-a558-468e-bf16-9fc9d0394abc)

Today there are at least a few good tools for experiment tracking and I would strongly recommend using one of them. They were **designed to treat machine learning experiments** as first-class citizens, and they will always:

- be **easier to use for a machine learning** person than general tools
- have **better integrations** with the ML ecosystem
- have **more experiment-focused features** than the general solutions  

**WANT TO EXPLORE THIS TOPIC?**[Read how to integrate Neptune into your codebase ![➡️](https://s.w.org/images/core/emoji/13.0.1/svg/27a1.svg)](https://docs.neptune.ai/getting-started/installation)

## Next steps

Experiment tracking is a practice even more than a tool or a logging method. It will take some time to really understand and implement:

- **what to keep track of** for your project, 
- **how to use that information** to improve future experiments, 
- how to **improve your teams’ unique workflow** with it,
- **when to even use** experiment tracking.

Hopefully, after reading this article, you have a good idea of whether experiment tracking can improve your (or your teams’) machine learning workflow. 

Do you want to start tracking your experiments?

- [Create a free Neptune account](https://neptune.ai/register)
- [Try Neptune on Colab](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/product-tours/how-it-works/showcase/Neptune-API-Tour.ipynb) (zero setup, no registration)
- [See the docs](https://docs.neptune.ai/)
- [Explore an example project](https://app.neptune.ai/common/example-project-tensorflow-keras/experiments?split=tbl&dash=charts&viewId=44675986-88f9-4182-843f-49b9cfa48599)

Are you hungry for more on the subject?

Here are some additional resources:

- Article: [A Complete Guide to Monitoring ML Experiments Live in Neptune](https://neptune.ai/blog/monitoring-machine-learning-experiments-guide)
- Docs page: [How to organize ML experimentation](https://docs.neptune.ai/how-to-guides/experiment-tracking/organize-ml-experiments)
- Docs page: [How to monitor ML runs live](https://docs.neptune.ai/how-to-guides/ml-run-monitoring/monitor-model-training-runs-live)
- Docs page: [How to connect Neptune to your codebase](https://docs.neptune.ai/getting-started/installation)
- Article: [How to Set Up Continuous Integration for Machine Learning with Github Actions and Neptune](https://neptune.ai/blog/continuous-integration-for-machine-learning-with-github-actions-and-neptune)

Happy experimenting!