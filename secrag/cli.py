"""secrag CLI entrypoint."""
import typer

app = typer.Typer(help="SEC filings RAG pipeline")


@app.command()
def smoke_test():
    """M0 verify: embed one string (dense + sparse) locally to confirm the stack works."""
    from fastembed import SparseTextEmbedding, TextEmbedding

    text = "Apple reported net income of $29.5 billion."
    dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    dense_vec = list(dense_model.embed([text]))[0]
    sparse_vec = list(sparse_model.embed([text]))[0]
    typer.echo(f"Dense vector dim={len(dense_vec)}")
    typer.echo(f"Sparse vector nnz={len(sparse_vec.indices)}")


@app.command()
def ingest(ticker: str, form: str = "10-K"):
    """Ingest a company's latest filing: text sections + XBRL facts, with provenance."""
    from secrag.ingest import ingest as run_ingest

    sections, facts = run_ingest(ticker, form)

    missing_prov = [s for s in sections if not s.provenance.item] + [
        f for f in facts if not f.provenance.concept
    ]
    if missing_prov:
        typer.echo(f"WARNING: {len(missing_prov)} items missing provenance", err=True)

    typer.echo(f"{ticker.upper()} {form}: {len(sections)} sections, {len(facts)} facts")
    for fct in facts:
        period = fct.period_end or "?"
        typer.echo(
            f"  {fct.provenance.concept:12s} {fct.value:>18,.0f} {fct.unit} "
            f"(period_end={period}, item provenance ok)"
        )


@app.command()
def ask(question: str):
    """Ask a question over the indexed filings. [M2]"""
    raise NotImplementedError("Implemented in M2")


@app.command("eval")
def run_eval(dataset: str = "financebench"):
    """Run the evaluation harness against a benchmark. [M3]"""
    raise NotImplementedError("Implemented in M3")


if __name__ == "__main__":
    app()
