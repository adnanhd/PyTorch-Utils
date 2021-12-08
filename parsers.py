
        with open(os.path.join('latex', name + ".tex"), 'w') as buf:
            latex = name.replace('_', '\\_')

            buf.write('\\documentclass{article}\n')
            buf.write('\\usepackage[utf8]{inputenc}\n')
            buf.write('\\usepackage{booktabs}\n')
            
            buf.write('\\title{}\n'.format("{" + latex + "}"))
            buf.write('\\author{adnan harun dogan, 2309896}\n')
            buf.write('\\date{December 2021}\n')

            buf.write('\\begin{document}\n')
            buf.write('\\maketitle\n')
            
            buf.write('\\section{Query 1}\n')
            pd.DataFrame(res[0, 0], columns=bins, index=grids).transpose().to_latex(buf)
            pd.DataFrame(res[0, 1], columns=bins, index=grids).transpose().to_latex(buf)


            buf.write('\\section{Query 2}\n')
            pd.DataFrame(res[1, 0], columns=bins, index=grids).transpose().to_latex(buf)
            pd.DataFrame(res[1, 1], columns=bins, index=grids).transpose().to_latex(buf)


            buf.write('\\section{Query 3}\n')
            pd.DataFrame(res[2, 0], columns=bins, index=grids).transpose().to_latex(buf)
            pd.DataFrame(res[2, 1], columns=bins, index=grids).transpose().to_latex(buf)
            buf.write('\\end{document}\n') 
       

        with open(os.path.join('html', name + ".html"), 'w') as bb:
            bb.write('<a href="index.html">go back</a><br>')
            
            pd.DataFrame(res[0, 0], columns=bins, index=grids).transpose().to_html(bb)
            pd.DataFrame(res[0, 1], columns=bins, index=grids).transpose().to_html(bb)

            pd.DataFrame(res[1, 0], columns=bins, index=grids).transpose().to_html(bb)
            pd.DataFrame(res[1, 1], columns=bins, index=grids).transpose().to_html(bb)

            pd.DataFrame(res[2, 0], columns=bins, index=grids).transpose().to_html(bb)
            pd.DataFrame(res[2, 1], columns=bins, index=grids).transpose().to_html(bb)
        
        f.write(f'<a href="{name}.html">{name}</a><br>')

    f.close()
