"""Paterson-Stockmeyer polynomial evaluation for ciphertexts."""


def evaluate_polynomial(engine, values, coefficients):
    """Evaluate a degree 2^r-1 polynomial on one or more ciphertexts."""
    coefficient_count = len(coefficients)
    settings = {
        8: (4, 2),
        16: (4, 4),
        32: (8, 4),
        64: (8, 8),
        128: (16, 8),
    }
    if coefficient_count not in settings:
        raise ValueError("supported polynomial degrees are 7, 15, 31, 63, and 127")

    single_value = not isinstance(values, list)
    if single_value:
        values = [values]

    giant_step_count, baby_step_size = settings[coefficient_count]
    results = []

    for value in values:
        powers = [None] * coefficient_count
        powers[1] = value
        for index in range(2, coefficient_count):
            if index <= baby_step_size or index & (index - 1) == 0:
                left = index // 2
                powers[index] = engine.multiply(powers[left], powers[index - left])

        baby_steps = []
        for giant_index in range(giant_step_count):
            block = coefficients[
                giant_index * baby_step_size : (giant_index + 1) * baby_step_size
            ]
            baby = engine.constant(block[0])

            # Splitting the last block reduces one level for larger polynomials.
            if coefficient_count >= 16 and giant_index == giant_step_count - 1:
                half = len(block) // 2
                for index in range(1, half):
                    baby = engine.add(baby, engine.multiply(block[index], powers[index]))

                upper = engine.constant(block[half])
                for index in range(half + 1, len(block)):
                    relative_index = index - half
                    if baby_step_size == 8 and index == 7:
                        term = engine.multiply(block[index], powers[1])
                        term = engine.multiply(term, powers[2])
                    else:
                        term = engine.multiply(block[index], powers[relative_index])
                    upper = engine.add(upper, term)
                baby = engine.add(baby, engine.multiply(upper, powers[half]))
            else:
                for index in range(1, len(block)):
                    baby = engine.add(
                        baby, engine.multiply(block[index], powers[index])
                    )
            baby_steps.append(baby)

        result = baby_steps[0]
        for giant_index in range(1, giant_step_count):
            exponent = giant_index * baby_step_size
            bit = 0
            while exponent:
                if exponent & 1:
                    baby_steps[giant_index] = engine.multiply(
                        baby_steps[giant_index], powers[2**bit]
                    )
                exponent >>= 1
                bit += 1
            result = engine.add(result, baby_steps[giant_index])
        results.append(result)

    return results[0] if single_value else results
