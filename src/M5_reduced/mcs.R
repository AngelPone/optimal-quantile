library(dplyr)
library(tidyr)

# MCS for M5 application

Methods <- c("QOpt($\\beta=100$)", "base", "ols", "wls", "shr", "sam")
Methods_name <- c("QOpt", "Base", "OLS", "MinT(var)", "MinT(shr)", "MinT(sam)")
ALPHAs <- c(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
output <- list()
for (scenario in c("normal", "skew")) {
  table_path <- paste0("output/logs/M5_ets_", scenario, "_insample.csv")
  tb <- data.table::fread(table_path)

  tbl_average <- tb |>
    group_by(alpha, method, window_idx) |>
    summarise(across(starts_with("series"), mean)) |>
    mutate(loss = rowMeans(pick(starts_with("series")))) |>
    select(-starts_with("series"))

  output[[scenario]] <- list()
  # mcs

  for (alpha in sort(unique(tbl_average$alpha))) {
    mcs_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      pivot_wider(names_from = "method", values_from = "loss") |>
      ungroup() |>
      select(-alpha, -window_idx) |>
      as.matrix() |>
      MCS::MCSprocedure(alpha = 0.05)
    included <- mcs_results@Info$included
    col_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      group_by(method) |>
      summarise(loss = mean(loss)) |>
      arrange(factor(method, levels = Methods)) |>
      pull(loss)
    output[[scenario]][[paste0(alpha)]] <- list(included = included, loss = col_results)
  }
}


for (scenario in c("normal", "skew")) {
  s_list <- output[[scenario]]
  total_text <- ""
  for (m_idx in seq_along(Methods)) {
    m_name <- Methods_name[m_idx]
    m <- Methods[m_idx]
    line_text <- paste0(m_name)
    for (alpha in ALPHAs) {
      val <- s_list[[paste0(alpha)]]$loss[m_idx]
      text <- sprintf("%.3f", val)
      if (min(s_list[[paste0(alpha)]]$loss) == val) {
        text <- sprintf("\\textbf{%s}", text)
      }
      if (m %in% s_list[[paste0(alpha)]]$included) {
        if (length(s_list[[paste0(alpha)]]$included) < length(Methods)) {
          text <- sprintf("%s\\textsuperscript{*}", text)
        }
      }
      line_text <- paste0(line_text, " & ", text)
    }
    if (m_idx < length(Methods)) {
      line_text <- paste0(line_text, "\\\\\n")
    }
    total_text <- paste0(total_text, line_text)
  }
  cat(total_text, file = sprintf("output/tables/M5_%s_insample.tex", scenario))
}

# insample
output <- list()
for (scenario in c("normal", "skew")) {
  table_path <- paste0("output/logs/M5_ets_", scenario, "_outsample.csv")
  tb <- data.table::fread(table_path)

  tbl_average <- tb |>
    group_by(alpha, method, window_idx) |>
    summarise(across(starts_with("series"), mean)) |>
    mutate(loss = rowMeans(pick(starts_with("series")))) |>
    select(-starts_with("series"))

  output[[scenario]] <- list()
  # mcs

  for (alpha in sort(unique(tbl_average$alpha))) {
    mcs_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      pivot_wider(names_from = "method", values_from = "loss") |>
      ungroup() |>
      select(-alpha, -window_idx) |>
      as.matrix() |>
      MCS::MCSprocedure(alpha = 0.05)
    included <- mcs_results@Info$included
    col_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      group_by(method) |>
      summarise(loss = mean(loss)) |>
      arrange(factor(method, levels = Methods)) |>
      pull(loss)
    output[[scenario]][[paste0(alpha)]] <- list(included = included, loss = col_results)
  }
}


for (scenario in c("normal", "skew")) {
  s_list <- output[[scenario]]
  total_text <- ""
  for (m_idx in seq_along(Methods)) {
    m_name <- Methods_name[m_idx]
    m <- Methods[m_idx]
    line_text <- paste0(m_name)
    for (alpha in ALPHAs) {
      val <- s_list[[paste0(alpha)]]$loss[m_idx]
      text <- sprintf("%.3f", val)
      if (min(s_list[[paste0(alpha)]]$loss) == val) {
        text <- sprintf("\\textbf{%s}", text)
      }
      if (m %in% s_list[[paste0(alpha)]]$included) {
        if (length(s_list[[paste0(alpha)]]$included) < length(Methods)) {
          text <- sprintf("%s\\textsuperscript{*}", text)
        }
      }
      line_text <- paste0(line_text, " & ", text)
    }
    if (m_idx < length(Methods)) {
      line_text <- paste0(line_text, "\\\\\n")
    }
    total_text <- paste0(total_text, line_text)
  }
  cat(total_text, file = sprintf("output/tables/M5_%s.tex", scenario))
}


m5 <- data.table::fread("output/logs/m5_weights.csv")
scenario <- "normal"
table_path <- paste0("output/logs/M5_ets_", scenario, "_outsample.csv")
tb <- data.table::fread(table_path)

by_levels <- NULL
for (level in sort(unique(m5$level))) {
  level_series <- m5 |>
    filter(level == .env$level) |>
    pull(idx)
  level_series <- level_series + 1
  level_series <- paste0("series", level_series)
  level_average <- tb |>
    select(window_idx, method, alpha, h, all_of(level_series)) |>
    mutate(loss = rowMeans(pick(starts_with("series")))) |>
    select(-starts_with("series")) |>
    filter(h %in% c(0, 13, 27)) |>
    group_by(window_idx, method, h) |>
    summarise(loss = mean(loss)) |>
    mutate(level = .env$level)
  by_levels <- rbind(by_levels, level_average)
}

output <- list()
for (h in c(0, 13, 27)) {
  tbl_average <- by_levels |> filter(h == .env$h)
  output[[paste0(h + 1)]] <- list()
  for (level in sort(unique(by_levels$level))) {
    mcs_results <- tbl_average |>
      filter(level == .env$level) |>
      pivot_wider(names_from = "method", values_from = "loss") |>
      ungroup() |>
      select(-level, -window_idx, -h) |>
      as.matrix() |>
      MCS::MCSprocedure(alpha = 0.05)
    included <- mcs_results@Info$included
    col_results <- tbl_average |>
      filter(level == .env$level) |>
      group_by(method) |>
      summarise(loss = mean(loss)) |>
      arrange(factor(method, levels = Methods)) |>
      pull(loss)
    output[[paste0(h + 1)]][[level]] <- list(included = included, loss = col_results)
  }
}


for (h in c(0, 13, 27)) {
  s_list <- output[[paste0(h + 1)]]
  total_text <- ""
  for (m_idx in seq_along(Methods)) {
    m_name <- Methods_name[m_idx]
    m <- Methods[m_idx]
    line_text <- paste0(m_name)
    for (level in paste0("level", 1:9)) {
      val <- s_list[[level]]$loss[m_idx]
      text <- sprintf("%.3f", val)
      if (min(s_list[[level]]$loss) == val) {
        text <- sprintf("\\textbf{%s}", text)
      }
      if (m %in% s_list[[level]]$included) {
        if (length(s_list[[level]]$included) < length(Methods)) {
          text <- sprintf("%s\\textsuperscript{*}", text)
        }
      }
      line_text <- paste0(line_text, " & ", text)
    }
    if (m_idx < length(Methods)) {
      line_text <- paste0(line_text, "\\\\\n")
    }
    total_text <- paste0(total_text, line_text)
  }
  cat(total_text, file = sprintf("output/tables/M5_%s.tex", paste0(h + 1)))
}
