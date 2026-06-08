import sys, io, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.WARNING)

from research_assistant.eval import run_eval, format_eval_report
results = run_eval(verbose=True)
if 'error' in results:
    print(f'ERROR: {results["error"]}')
else:
    report = format_eval_report(results)
    print(report)
    # 保存到文件
    with open('docs/evaluation.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print('\n报告已保存到 docs/evaluation.md')
