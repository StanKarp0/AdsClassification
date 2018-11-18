import pandas as pd


def count_table():
    categories = pd.read_csv('categories.csv')

    for i, cat in categories.iterrows():
        print("\hline\n%d & %s & %s & %d\\\\" %
              (i,
               cat['text'].replace('_', ' '),
               cat['description'].replace('_', ' '),
               cat['count']))
    print('\hline')


def example_table():
    categories = pd.read_csv('categories.csv')

    print("""\\begin{center}
  \\begin{longtable}{ | m{5cm} | c | m{0.5cm} |}
    \hline
    Kategoria & Przykłady & N
    \\\\ \hline""")

    for i, cat in categories.iterrows():
        print("""    \\begin{minipage}[t]{5cm}
        %s
    \\begin{itemize}
        \item %s
      \end{itemize}
      \end{minipage}
     &
    \\begin{minipage}{.6\\textwidth}
      \includegraphics[width=\linewidth, height=50mm]{%s.png}
    \end{minipage}
    &
    \\begin{minipage}[t]{0.5cm}
    %d
      \end{minipage}
    \\\\ \hline""" % (
            cat['text'].replace('_', ' '),
            cat['description'],
            cat['text'],
            cat['count']
        ))


    print("""  \end{longtable}
  \caption{Kategorie}\label{tbl:categories}
\end{center}""")


if __name__ == '__main__':
    example_table()
    # count_table()